from argparse import Namespace, ArgumentParser
import csv
from logging import Logger, disable
import os
from pprint import pformat
from typing import List
import pandas as pd
import numpy as np
from torch.utils.data.dataset import ConcatDataset
from tensorboardX import SummaryWriter
import torch
from tqdm import trange
import pickle
from torch.optim.lr_scheduler import ExponentialLR
import random
import sys

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from chemprop.data import StandardScaler, MoleculeDataset, MoleculeDatapoint
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop.models import build_model
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint
from chemprop.parsing import add_predict_args
from .make_predictions import make_predictions
from .make_predictions_atomic_unc import make_predictions_atomic_unc

def run_training_atl(args: Namespace, logger: Logger = None, active_iter: int = -1, logger_all: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    # set logger and logger_all
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        raise ValueError(f'Logger individual problem: active iter:{active_iter}')
    if logger_all is not None:
        debug_all, info_all = logger_all.debug, logger_all.info
    else:
        raise ValueError('Logger_all problem')

    # Set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # Print args
    debug(pformat(vars(args)))

    # Split data
    debug(f'Splitting data with seed {args.seed}')

    # init dataset
    debug(f'Active learning, iter: {active_iter}')
    debug_all(f'Active learning, iter: {active_iter}')
    # initiate dataset
    if active_iter == 0: 
        # Get data
        debug('Loading data')
        args.task_names = get_task_names(args.data_path)
        data = get_data(path=args.data_path, args=args, logger=logger)
        args.num_tasks = data.num_tasks()
        args.features_size = data.features_size()
        debug(f'Number of tasks = {args.num_tasks}, type = {type(args.num_tasks)}.')
        train_data, remain_data, test_data = split_data(data=data, split_type=args.split_type, sizes=(0.15, 0.75, 0.1), seed=args.seed, args=args, logger=logger)
        train_data, val_data, _ = split_data(data=train_data, split_type=args.split_type, sizes=(0.9, 0.1, 0.0), seed=args.seed, args=args, logger=logger)
        args.k_samples = len(remain_data) // 6
        # read data from data_path
        with open(args.data_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

            lines_by_smiles = {}
            indices_by_smiles = {}
            for i, line in enumerate(reader):
                smiles = line[0]
                lines_by_smiles[smiles] = line
                indices_by_smiles[smiles] = i
        # write data from data_saved
        makedirs(os.path.join(args.active_dir, 'saved_data'))
        all_split_indices = []
        for dataset, name in [(train_data, 'train'), (val_data, 'val'), (remain_data, 'remain')]:
            with open(os.path.join(args.active_dir, 'saved_data', name + '_full.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for smiles in dataset.smiles():
                    writer.writerow(lines_by_smiles[smiles])
            split_indices = []
            for smiles in dataset.smiles():
                split_indices.append(indices_by_smiles[smiles])
            split_indices = sorted(split_indices)
            all_split_indices.append(split_indices)
        # save test data to save_dir
        with open(os.path.join(args.save_dir, 'test_full.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for smiles in test_data.smiles():
                writer.writerow(lines_by_smiles[smiles])
            split_indices = []
            for smiles in test_data.smiles():
                split_indices.append(indices_by_smiles[smiles])
            split_indices = sorted(split_indices)
            all_split_indices.append(split_indices)
        with open(os.path.join(args.active_dir, 'split_indices.pckl'), 'wb') as f:
            pickle.dump(all_split_indices, f)
        
        
    # load data from current folder (e.g. active iter 1 load data from active iter 1)
    else:
        args.write_true_val = False  
        args.data_path_iter = os.path.join(args.active_dir, 'saved_data')
        info(f'loading data from: {args.data_path_iter}')
        # load train_data, remain data, test_data
        try:
            args.task_names
            args.k_samples
        except:
            args.task_names = ['Hf_0']
            # args.k_samples = 6
            args.k_samples = 98121 // 6
        train_data = get_data(path=os.path.join(args.data_path_iter, 'train_full.csv'), args=args, logger=logger)
        val_data = get_data(path=os.path.join(args.data_path_iter, 'val_full.csv'), args=args, logger=logger)
        remain_data = get_data(path=os.path.join(args.data_path_iter, 'remain_full.csv'), args=args, logger=logger)
        test_data = get_data(path=os.path.join(args.save_dir, 'test_full.csv'), args=args, logger=logger)
        args.num_tasks = train_data.num_tasks()
        args.features_size = train_data.features_size()
        debug(f'Number of tasks = {args.num_tasks}, type = {type(args.num_tasks)}')


    if args.dataset_type == 'classification':
        raise ValueError('Classification task is not provided.')

    if args.features_scaling:  # false
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)
    
    debug(f'Train size = {len(train_data):,} | Val size = {len(val_data):,} | Test size = {len(test_data):,} | Remain size = {len(remain_data):,}')
    debug_all(f'Train size = {len(train_data):,} | Val size = {len(val_data):,} | Test size = {len(test_data):,} | Remain size = {len(remain_data):,}')
    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        train_smiles, train_targets = train_data.smiles(), train_data.targets()

        if args.y_scaling:  # devide with training data standard deviation
            debug(f'{args.fp_method} scale y value')
            debug_all(f'{args.fp_method} scale y value')
            scaler = StandardScaler().fit(train_targets)
            scaled_targets = scaler.transform(train_targets).tolist()
        else:  # without scaling
            debug(f'{args.fp_method} unscale y value')
            debug_all(f'{args.fp_method} unscale y value')
            scaler = None
            scaled_targets = train_targets
            
        train_data.set_targets(scaled_targets)
    else:
        scaler = None

    # Get loss and metric functions
    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)
    debug_all(f'loss_func: {loss_func} | metric_func: {metric_func}')

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        model_dir = os.path.join(args.active_dir, 'model', f'model_{model_idx}')
        makedirs(model_dir)
        try:
            writer = SummaryWriter(log_dir=model_dir)
        except:
            writer = SummaryWriter(logdir=model_dir)
        # Load/build model
        if args.checkpoint_paths is not None: # transfer learning
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            debug_all(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], current_args=args, logger=logger)
        else:  # train
            debug(f'Building model {model_idx}')
            debug_all(f'Building model {model_idx}')
            model = build_model(args)
        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')

        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(model_dir, 'model.pt'), model, scaler, features_scaler, args)

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        early_stopping_step = 0
        early_stopping = 15
        for epoch in range(args.epochs):
            debug(f'Epoch {epoch}')

            n_iter = train(
                model=model,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            val_scores, val_rmses, val_maes = evaluate(
                model=model,
                data=val_data,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger,
                sampling_size=args.sampling_size,
                fp_method=args.fp_method
            )

            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            avg_val_rmse = np.nanmean(val_rmses)
            avg_val_mae = np.nanmean(val_maes)
            debug(f'Validation {args.metric} = {avg_val_score:.6f} | Validation rmse = {avg_val_rmse:.6f} | Validation mae = {avg_val_mae:.6f}')
            writer.add_scalar(f'validation_{args.metric}', avg_val_score, n_iter)
            writer.add_scalar('validation_rmse', avg_val_rmse, n_iter)
            writer.add_scalar('validation_mae', avg_val_mae, n_iter)
            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {args.metric} = {avg_val_score:.6f} | Validation rmse = {avg_val_rmse:.6f} | Validation mae = {avg_val_mae:.6f}')
                    writer.add_scalar(f'validation_{task_name}_{args.metric}', val_score, n_iter)

            # Save model checkpoint if improved validation score
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                early_stopping_step = 0
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(model_dir, 'model.pt'), model, scaler, features_scaler, args)
            else:
                early_stopping_step += 1
            
            # break if early stopping happens
            if early_stopping_step == early_stopping: 
                debug(f'STOPPING CONDITION IS MET!! epoch:{epoch}')
                break
        
        # Evaluate on test set using model with best validation score
        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(model_dir, 'model.pt'), cuda=args.cuda, logger=logger)
        
        
        test_preds, test_ales, _, _, _ = predict(
            model=model,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler,
            sampling_size=args.sampling_size,
            fp_method=args.fp_method
        )
        test_scores, test_rmse, test_mae = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            ################ add aleatoric to evaluate metrics  ###################
            ales=test_ales,           
            ################ add aleatoric to evaluate metrics  ###################
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            logger=logger
        )   

        # Average test score
        avg_test_score = np.nanmean(test_scores)
        avg_test_rmse = np.nanmean(test_rmse)
        avg_test_mae = np.nanmean(test_mae)
        info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f} | test rmse = {avg_test_rmse:.6f} | test mae = {avg_test_mae:.6f}')
        writer.add_scalar(f'test_{args.metric}', avg_test_score, 0)

        if args.show_individual_scores:
            # Individual test scores
            for task_name, test_score in zip(args.task_names, test_scores):
                info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f} | test rmse = {avg_test_rmse:.6f} | test mae = {avg_test_mae:.6f}')
                writer.add_scalar(f'test_{task_name}_{args.metric}', test_score, n_iter)

    
    # evaluate ensemble on test set
    assert args.checkpoint_paths is None
    args.checkpoint_paths = []
    for root, _, files in os.walk(args.active_dir):  # root, dirs, files
        for fname in files:
            if fname.endswith('.pt'):
                args.checkpoint_paths.append(os.path.join(root, fname))
    all_preds = np.zeros((len(test_data), len(args.checkpoint_paths)))
    all_ale_uncs = np.zeros((len(test_data), len(args.checkpoint_paths)))
    
    if (args.active_uncertainty is not None) and args.atomic_unc:
        all_atomic_preds = np.zeros((len(test_data), args.max_atom_size, len(args.checkpoint_paths)))
        all_atomic_ales = np.zeros((len(test_data), args.max_atom_size, len(args.checkpoint_paths)))
    for index, checkpoint_path in enumerate(args.checkpoint_paths):
        # Load model
        print(f'loading data from checkpoint_path {index}: {checkpoint_path}')
        model = load_checkpoint(checkpoint_path, current_args=args, cuda=args.cuda)
        model_preds, ale_uncs, _, atomic_preds, atomic_uncs = predict(
            model=model,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler,
            sampling_size=args.sampling_size,
            fp_method=args.fp_method,
            atomic_unc=args.atomic_unc
        )
 
        all_preds[:, index] = np.array(model_preds).squeeze() # (num_mols, 1) -> (num_mols)
        all_ale_uncs[:, index] = np.array(ale_uncs).squeeze()
        if (args.active_uncertainty is not None) and args.atomic_unc:
            all_atomic_preds[:, :, index] = atomic_preds  # num_mols x atom_preds x models
            all_atomic_ales[:, :, index] = atomic_uncs    
        else:
            assert (atomic_preds is None) and (atomic_uncs is None)

    
    # Evaluate ensemble on test set
    avg_test_preds = (np.sum(all_preds, axis=1) / len(args.checkpoint_paths))[:, np.newaxis].tolist()
    avg_test_ales = (np.sum(all_ale_uncs, axis=1) / len(args.checkpoint_paths))[:, np.newaxis].tolist()
    avg_test_epis = np.var(all_preds, axis=1)[:, np.newaxis].tolist()
    if (args.active_uncertainty is not None) and args.atomic_unc:
        avg_test_atomic_ales = np.sum(all_atomic_ales, axis=2) / len(args.checkpoint_paths)
        avg_test_atomic_epis = np.var(all_atomic_preds, axis=2)
        avg_test_atomic_total = avg_test_atomic_ales + avg_test_atomic_epis  # mol x max_atom_size
        avg_test_atomic_max_ales = np.max(avg_test_atomic_ales, axis=1)  # take max ale_unc of atoms in a molecule
        avg_test_atomic_max_epis = np.max(avg_test_atomic_epis, axis=1)
        avg_test_atomic_max_total = np.max(avg_test_atomic_total, axis=1)

    ensemble_scores, ensemble_rmse, ensemble_mae = evaluate_predictions(
        preds=avg_test_preds,
        targets=test_targets,
        ################ add aleatoric to evaluate metrics  ###################
        ales=avg_test_ales,                 
        ################ add aleatoric to evaluate metrics  ###################
        num_tasks=args.num_tasks,
        metric_func=metric_func,
        dataset_type=args.dataset_type,
        logger=logger
    )
    
    # Average ensemble score
    avg_ensemble_test_score = np.nanmean(ensemble_scores)
    avg_ensemble_test_rmse = np.nanmean(ensemble_rmse)
    avg_ensemble_test_mae = np.nanmean(ensemble_mae)
    writer.add_scalar(f'ensemble_test_{args.metric}', avg_ensemble_test_score, 0)

    # save testing results in current saved_model folder
    if (args.active_uncertainty is not None) and args.atomic_unc:
        test_output = np.hstack((np.array(test_data.smiles())[:, np.newaxis], np.array(test_data.targets()), np.array(avg_test_preds), \
            np.array(avg_test_ales), np.array(avg_test_epis), np.array(avg_test_ales) + np.array(avg_test_epis), avg_test_atomic_max_ales[:, np.newaxis], avg_test_atomic_max_epis[:, np.newaxis], avg_test_atomic_max_total[:, np.newaxis]))
        # test_output = test_output[np.argsort(test_output[:, type_unc_sort[args.active_uncertainty]].astype(float))[::-1]]
        assert test_output.shape == (test_data.__len__(), 9)
        pd.DataFrame(test_output, columns=['smiles', 'true', 'pred', 'ale_unc', 'epi_unc', 'total_unc', 'max_atom_ale_unc', 'max_atom_epi_unc', 'max_atom_total_unc']).to_csv(os.path.join(args.active_dir, 'saved_data', 'test_pred.csv'), index=False)
        del test_output, avg_test_atomic_ales, avg_test_atomic_epis, avg_test_atomic_total
    else:
        test_output = np.hstack((np.array(test_data.smiles())[:, np.newaxis], np.array(test_data.targets()), np.array(avg_test_preds), np.array(avg_test_ales), np.array(avg_test_epis), np.array(avg_test_ales) + np.array(avg_test_epis)))
        # test_output = test_output[np.argsort(test_output[:, type_unc_sort[args.active_uncertainty]].astype(float))[::-1]]
        assert test_output.shape == (test_data.__len__(), 6)
        pd.DataFrame(test_output, columns=['smiles', 'true', 'pred', 'ale_unc', 'epi_unc', 'total_unc']).to_csv(os.path.join(args.active_dir, 'saved_data', 'test_pred.csv'), index=False)
        del test_output, all_preds, all_ale_uncs
    if active_iter == 6:
        return avg_ensemble_test_score, avg_ensemble_test_rmse, avg_ensemble_test_mae

    args.test_path = os.path.join(args.active_dir, 'saved_data', 'remain_full.csv')
    args.preds_path = os.path.join(args.active_dir, 'saved_data', 'remain_pred.csv')
    args.checkpoint_dir = args.active_dir # model path
    args.estimate_variance = True
    args.write_true_val = True
    args.checkpoint_paths = []
    for root, _, files in os.walk(args.checkpoint_dir):
        for fname in files:
            if fname.endswith('.pt'):
                args.checkpoint_paths.append(os.path.join(root, fname))

    args.ensemble_size = len(args.checkpoint_paths)

    debug_all(f'Predict remaining data with {args.ensemble_size} models from {args.checkpoint_dir}')

    if args.ensemble_size == 0:
        raise ValueError(f'Failed to find any model checkpoints in directory "{args.checkpoint_dir}"')

    if (args.active_uncertainty is not None) and args.atomic_unc:
        debug(f'predicting atomic unc of remaining data...')
        make_predictions_atomic_unc(args)
    else:
        make_predictions(args)

    # next iter file
    args.active_dir_next = os.path.join(args.save_dir, f'active_iter{active_iter+1}', 'saved_data')
    makedirs(args.active_dir_next)

    load_remain_data = pd.read_csv(os.path.join(args.active_dir, 'saved_data', 'remain_pred.csv')).values

    if args.active_uncertainty == 'random':
        np.random.seed(args.seed)
        np.random.shuffle(load_remain_data)
    else:
        if args.atomic_unc:
            type_unc_sort = {'aleatoric': 6, 'epistemic': 7, 'total': 8}
            debug_all(f'sorting remain data by atomic {args.active_uncertainty} uncertainty...')
        else:
            type_unc_sort = {'aleatoric': 3, 'epistemic': 4, 'total': 5}
            debug_all(f'sorting remain data by molecular {args.active_uncertainty} uncertainty...')
        load_remain_data = load_remain_data[np.argsort(load_remain_data[:, type_unc_sort[args.active_uncertainty]].astype(float))[::-1]]

    kremain_data = MoleculeDataset([MoleculeDatapoint(line=line, args=args,) for i, line in enumerate(load_remain_data[:args.k_samples, :2].tolist())])
    load_remain_data = MoleculeDataset([MoleculeDatapoint(line=line, args=args,) for i, line in enumerate(load_remain_data[args.k_samples:, :2].tolist())])

    # combine train, val, k-remain
    # inverse transform targets of training data 
    if scaler is not None:
        debug(f'{args.fp_method} with scaler. inverse transform targets of training data.')
        train_data.set_targets(scaler.inverse_transform(train_data.targets()))
    concat_train_val_kremain = []
    for i in [train_data, val_data, kremain_data]:
        concat_train_val_kremain.extend(i)
    concat_train_val_kremain = MoleculeDataset(concat_train_val_kremain)

    info(f'train-val-k_remain size: {len(concat_train_val_kremain):,} | rest_remain_data size: {len(load_remain_data):,}.')

    train_data, val_data, _ = split_data(data=concat_train_val_kremain, split_type='random', sizes=(0.9, 0.1, 0.0), seed=args.seed, args=args, logger=logger)

    for dataset, name in [(train_data, 'train'), (val_data, 'val'), (load_remain_data, 'remain')]:
        data_output = np.hstack((np.array(dataset.smiles())[:, np.newaxis], np.array(dataset.targets())))
        pd.DataFrame(data_output, columns=['smiles', 'true']).to_csv(os.path.join(args.active_dir_next, name + '_full.csv'), index=False)
    debug(f'active learning iter: {active_iter} saved new dataset at : {args.active_dir_next}/saved_data.')    
    debug_all(f'active learning iter: {active_iter} saved new dataset at : {args.active_dir_next}/saved_data.')
    
    args.checkpoint_paths = args.checkpoint_dir = None  # turn off checkpoint_paths for preventing transfer learning in the next round

    return avg_ensemble_test_score, avg_ensemble_test_rmse, avg_ensemble_test_mae
