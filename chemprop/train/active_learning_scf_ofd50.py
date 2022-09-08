from argparse import Namespace
from logging import Logger
import os
import csv
from typing import Tuple

import numpy as np

from .run_training_atl_scf_ofd50 import run_training_atl_scf_ofd50
from chemprop.data.utils import get_task_names
from chemprop.utils import makedirs


def active_learning_scf_ofd50(args: Namespace, logger: Logger = None, active_iter: int = 0, logger_all: Logger = None) -> Tuple[float, float]:
    """k-fold cross validation"""
    info = logger.info if logger is not None else print
    info_all = logger_all.info if logger_all is not None else print

    # Initialize relevant variables
    save_dir = args.save_dir

    # only one fold 
    args.active_dir = os.path.join(save_dir, f'active_iter{active_iter}')
    makedirs(args.active_dir)

    # test logger_all
    model_scores, model_rmse, model_mae = run_training_atl_scf_ofd50(args, logger, active_iter, logger_all)

    info_all(f'active learning iter: {active_iter}, test {args.metric}: {model_scores:.6f}, test rmse: {model_rmse:.6f}, test_mae: {model_mae:.6f}\n')
    info(f'active learning iter: {active_iter}, test {args.metric}: {model_scores:.6f}, test rmse: {model_rmse:.6f}, test mae: {model_mae:.6f}\n')

    active_log = open(f'{save_dir}/active_log.csv', 'a')
    writer = csv.writer(active_log)
    writer.writerow([active_iter, round(model_scores, 5), round(model_rmse, 5), round(model_mae, 5)])
    active_log.close()

