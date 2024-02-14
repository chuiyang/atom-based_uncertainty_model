import logging
import math
import os
from typing import Callable, List, Tuple, Union
from argparse import Namespace

from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from chemprop.data import StandardScaler
from chemprop.models import build_model, MoleculeModel
from chemprop.nn_utils import NoamLR, InverseLR
import numpy as np

def makedirs(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. isfiled == True),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


def save_checkpoint(path: str,
                    model: MoleculeModel,
                    scaler: StandardScaler = None,
                    features_scaler: StandardScaler = None,
                    args: Namespace = None):
    """
    Saves a model checkpoint.

    :param model: A MoleculeModel.
    :param scaler: A StandardScaler fitted on the data.
    :param features_scaler: A StandardScaler fitted on the features.
    :param args: Arguments namespace.
    :param path: Path where checkpoint will be saved.
    """
    state = {
        'args': args,
        'state_dict': model.state_dict(),
        'data_scaler': {
            'means': scaler.means,
            'stds': scaler.stds
        } if scaler is not None else None,
        'features_scaler': {
            'means': features_scaler.means,
            'stds': features_scaler.stds
        } if features_scaler is not None else None
    }
    torch.save(state, path)


def load_checkpoint(path: str,
                    current_args: Namespace = None,
                    cuda: bool = None,
                    logger: logging.Logger = None) -> MoleculeModel:
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :param cuda: Whether to move model to cuda.
    :param logger: A logger.
    :return: The loaded MoleculeModel.
    """
    debug = logger.debug if logger is not None else print

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']

    if current_args is not None:  
        try: 
            if current_args.pred_max_atom_size is not None and current_args.pred_max_atom_size != args.max_atom_size:  # prediction process
                debug(f'{path}: generalization prediction, original max_atom_size: {args.max_atom_size}, now predicting max_atom_size = {current_args.pred_max_atom_size} molecule.')
            else:
                debug(f'{path}: prediction process, but without generalization, use the original max_atom_size: {args.max_atom_size}')
            args = current_args
            args.max_atom_size = current_args.pred_max_atom_size
        except:
            if current_args.max_atom_size is not None:
                debug(f'transfer learning: change max_atom_size from {args.max_atom_size} to {current_args.max_atom_size}')
                args.max_atom_size = current_args.max_atom_size
            else:
                debug('chemprop/utils/86 current args.max_atom_size is None')

    args.cuda = cuda if cuda is not None else args.cuda

    # Build model
    model = build_model(args)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():

        if param_name not in model_state_dict:
            debug(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
        elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
            debug(f'Pretrained parameter "{param_name}" '
                  f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                  f'model parameter of shape {model_state_dict[param_name].shape}.')
        else:
            # debug(f'Loading pretrained parameter "{param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if cuda:
        debug('Moving model to cuda')
        model = model.cuda()

    return model


def load_scalers(path: str) -> Tuple[StandardScaler, StandardScaler]:
    """
    Loads the scalers a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: A tuple with the data scaler and the features scaler.
    """
    state = torch.load(path, map_location=lambda storage, loc: storage)

    scaler = StandardScaler(state['data_scaler']['means'],
                            state['data_scaler']['stds']) if state['data_scaler'] is not None else None
    features_scaler = StandardScaler(state['features_scaler']['means'],
                                     state['features_scaler']['stds'],
                                     replace_nan_token=0) if state['features_scaler'] is not None else None

    return scaler, features_scaler


def load_args(path: str) -> Namespace:
    """
    Loads the arguments a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: The arguments Namespace that the model was trained with.
    """
    return torch.load(path, map_location=lambda storage, loc: storage)['args']


def load_task_names(path: str) -> List[str]:
    """
    Loads the task names a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: The task names that the model was trained with.
    """
    return load_args(path).task_names


def heteroscedastic_loss(true, mean, var):
    """
    Compute the heteroscedastic loss for regression.

    :param true: A list of true values.
    :param mean: A list of means (output predictions).
    :param var: A list of vars (predicted variances).
    :return: Computed loss.
    """
    loss = (var**(-1)) * (true - mean)**2 + torch.log(var)
    return loss

def heteroscedastic_loss_mol(true, mean, log_var):
    """
    Compute the heteroscedastic loss for regression.

    :param true: A list of true values.
    :param mean: A list of means (output predictions).
    :param log_var: A list of logvars (log of predicted variances).
    :return: Computed loss.
    """
    precision = torch.exp(-log_var)
    loss = precision * (true - mean)**2 + log_var
    return loss


def heteroscedastic_metric(true, mean, var):  # var = ale_unc
    """
    Compute the heteroscedastic loss for regression.

    :param true: A list of true values.
    :param mean: A list of means (output predictions).
    :param var: A list of vars (predicted variances).
    :return: Computed loss.
    """
    true, mean, var = np.array(true).astype(float), np.array(mean).astype(float), np.array(var).astype(float)
    loss = np.mean((var**(-1)) * (true - mean)**2 + np.log(var))
    return loss


def get_loss_func(args: Namespace) -> nn.Module:
    """
    Gets the loss function corresponding to a given dataset type.

    :param args: Namespace containing the dataset type ("classification" or "regression").
    :return: A PyTorch loss function.
    """
    if args.dataset_type == 'classification':
        return nn.BCEWithLogitsLoss(reduction='none')
    
    if args.dataset_type == 'multiclass':
        return nn.CrossEntropyLoss(reduction='none')

    ### For mixed model ###   
    if args.dataset_type == 'regression' and args.aleatoric and (args.fp_method in ['atomic', 'hybrid_dim0', 'hybrid_dim1']):
        return heteroscedastic_loss

    elif args.dataset_type == 'regression' and args.aleatoric and args.fp_method == 'molecular':
        return heteroscedastic_loss_mol
    ### For mixed model ###
    
    if args.dataset_type == 'regression':
        return nn.MSELoss(reduction='none')
    raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')


def prc_auc(targets: List[int], preds: List[float]) -> float:
    """
    Computes the area under the precision-recall curve.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def rmse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    return math.sqrt(mean_squared_error(targets, preds))


def mse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed mse.
    """
    return mean_squared_error(targets, preds)


def accuracy(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the accuracy of a binary prediction task using a given threshold for generating hard predictions.
    Alternatively, compute accuracy for a multiclass prediction task by picking the largest probability. 

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed accuracy.
    """
    if type(preds[0]) == list: # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds] # binary prediction
    return accuracy_score(targets, hard_preds)


def get_metric_func(metric: str) -> Callable[[Union[List[int], List[float]], List[float]], float]:
    """
    Gets the metric function corresponding to a given metric name.

    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a list of predictions and returns.
    """
    if metric == 'auc':
        return roc_auc_score

    if metric == 'prc-auc':
        return prc_auc

    if metric == 'rmse':
        return rmse
    
    if metric =='mse':
        return mse

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'r2':
        return r2_score
    
    if metric == 'accuracy':
        return accuracy
    
    if metric == 'cross_entropy':
        return log_loss

    if metric == 'heteroscedastic':
        return heteroscedastic_metric

    raise ValueError(f'Metric "{metric}" not supported.')


def build_optimizer(model: nn.Module, args: Namespace) -> Optimizer:
    """
    Builds an Optimizer.

    :param model: The model to optimize.
    :param args: Arguments.
    :return: An initialized Optimizer.
    """
    params = [{'params': model.parameters(), 'lr': args.init_lr, 'weight_decay': 0}]  # origin

    return Adam(params)  # origin


def build_optimizer_multimodel(models_dict: dict, args: Namespace, names_to_release: List[str]) -> Optimizer:
    """
    Builds an Optimizer.

    :param model: The models to optimize.
    :param args: Arguments.
    :return: An initialized Optimizer.
    """
    param = list(p for name, p in models_dict[f'model_0'].named_parameters() if name in names_to_release)
    for i in range(1, len(models_dict.keys())):
        param += list(p for name, p in models_dict[f'model_{i}'].named_parameters() if name in names_to_release)
    params = [{'params': param, 'lr': args.init_lr, 'weight_decay': 0}]  # origin
    return AdamW(params)  # origin


def build_lr_scheduler(optimizer: Optimizer, args: Namespace, total_epochs: List[int] = None) -> _LRScheduler:
    """
    Builds a learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: Arguments.
    :param total_epochs: The total number of epochs for which the model will be run.
    :return: An initialized learning rate scheduler.
    """
    # Learning rate scheduler
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=total_epochs or [args.epochs] * args.num_lrs,
        steps_per_epoch=args.train_data_size // args.batch_size,
        init_lr=[args.init_lr],
        max_lr=[args.max_lr],
        final_lr=[args.final_lr]
    )


def build_lr_scheduler_inverse(optimizer: Optimizer, args: Namespace, total_epochs: List[int] = None) -> _LRScheduler:
    """
    Builds a learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: Arguments.
    :param total_epochs: The total number of epochs for which the model will be run.
    :return: An initialized learning rate scheduler.
    """
    # Learning rate scheduler
    return InverseLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=total_epochs or [args.epochs] * args.num_lrs,
        steps_per_epoch=args.train_data_size // args.batch_size,
        init_lr=[args.init_lr],
        max_lr=[args.max_lr],
        final_lr=[args.final_lr]
    )


def create_logger(name: str, save_dir: str = None, quiet: bool = False, active_iter: int = -1) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e. print only important info).
    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # default True, False: message will not send to ancient loggers

    # Set logger depending on desired verbosity (default level: warning)
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG) 
    logger.addHandler(ch)

    if save_dir is not None:
        if active_iter != -1:
            save_dir = os.path.join(save_dir, f'active_iter{active_iter}')
        makedirs(save_dir)

        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger

def create_logger_atl_all(name: str, save_dir: str = None, quiet: bool = False, active_iter: int = -1) -> logging.Logger:
    """
    Creates a logger for all active learning process. need only debug file for all log.

    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e. print only important info).
    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # default True, False: message will not send to ancient loggers

    # Set logger depending on desired verbosity (default level: warning)
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG) 
    logger.addHandler(ch)

    if save_dir is not None:
        if active_iter != -1:
            save_dir = os.path.join(save_dir, f'active_iter{active_iter}')
        makedirs(save_dir)

        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose_atl_all.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet_atl_all.log'))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger

def transfer_learning_check(model, freeze_GCNN, logger):
    debug = logger.debug if logger is not None else print
    debug(f'freeze encoder layer: {freeze_GCNN}')
    if not freeze_GCNN:
        return model
    else:
        for name, param in model.named_parameters():
            if 'encoder' in name:
                param.requires_grad = False
            debug(f'{name}, grad = {param.requires_grad}')
        return model

def transfer_learning_release_varlayer(model, freeze_GCNN, model_idx, logger):
    debug = logger.debug if logger is not None else print
    debug(f'freeze encoder layer: {freeze_GCNN}')
    if not freeze_GCNN:
        return model
    else:
        names_to_release = []
        for name, param in model.named_parameters():
            if ('std_layer' not in name) and ('logvar_layer' not in name):
                param.requires_grad = False
            # elif model_idx not in [0, 1, 2]:
            #     param.requires_grad = False
            else:
                # initialize params : https://zhuanlan.zhihu.com/p/53712833
                # debug(f'param.dim() : {param.dim()}')
                # if param.dim() == 1:
                #     nn.init.constant_(param, 0)
                # else:
                #     nn.init.xavier_normal_(param)
                names_to_release.append(name)
            debug(f'{name}, grad = {param.requires_grad}, {param.shape}')
        return model, name