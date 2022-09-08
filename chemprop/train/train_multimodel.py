from argparse import Namespace
import logging
from typing import Callable, List, Union

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import tqdm 

from chemprop.data import MoleculeDataset
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR, InverseLR


def train_multimodel(models_dict: dict,
          data: Union[MoleculeDataset, List[MoleculeDataset]],
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: Namespace,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: A MoleculeDataset (or a list of MoleculeDatasets if using moe).
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print
    
    # model.train()
    
    data.shuffle()

    loss_sum, iter_count = 0, 0

    num_iters = len(data) // args.batch_size * args.batch_size  # don't use the last batch if it's small, for stability

    iter_size = args.batch_size

    for i in range(0, num_iters, iter_size):
        # Prepare batch
        if i + args.batch_size > len(data):
            break
        mol_batch = MoleculeDataset(data[i:i + args.batch_size])
        smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
        batch = smiles_batch
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])
        means_multi = torch.zeros(targets.shape[0], 1, args.ensemble_size)
        logvars_multi = torch.zeros(targets.shape[0], 1, args.ensemble_size)
        if args.cuda:
            mask, targets, means_multi, logvars_multi = mask.cuda(), targets.cuda(), means_multi.cuda(), logvars_multi.cuda()

        class_weights = torch.ones(targets.shape)
        
        if args.cuda:
            class_weights = class_weights.cuda()

        # Run model
        for index in range(len(models_dict.keys())):
            model = models_dict[f'model_{index}']
            model.train()
            # model.zero_grad()
        
            if not args.aleatoric:
                preds = model(batch, features_batch)
                if args.dataset_type == 'multiclass':   
                    targets = targets.long()
                    loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * class_weights * mask
                else:
                    loss = loss_func(preds, targets) * class_weights * mask
            else:
                means, logvars, _, _ = model(batch, features_batch)
                means_multi[:, :, index] = means
                logvars_multi[:, :, index] = logvars

        means = torch.mean(means_multi, 2)
        logvars = torch.mean(logvars_multi, 2)

        loss = loss_func(targets, means, logvars) * class_weights * mask
        
        loss = loss.sum() / mask.sum()

        if args.epistemic == 'mc_dropout':
            reg_loss = args.reg_acc.get_sum()
            loss += reg_loss
        
        loss_sum += loss.item()
        iter_count += len(mol_batch)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if isinstance(scheduler, NoamLR):  # class
            scheduler.step()

        n_iter += len(mol_batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum, iter_count = 0, 0

            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, lr = {lrs_str}')

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)
    return n_iter
