"""Trains a model on a dataset."""

from chemprop.parsing import parse_train_args
from chemprop.train import active_learning_ofd
from chemprop.utils import create_logger, create_logger_atl_all

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    args = parse_train_args()
    logger_all = create_logger_atl_all(name='train_atl', save_dir=args.save_dir, quiet=True)
    for active_iter in range(5, 7):
        logger = create_logger(name=f'train_atl_{active_iter}', save_dir=args.save_dir, quiet=args.quiet, active_iter=active_iter)
        active_learning_ofd(args, logger, active_iter=active_iter, logger_all=logger_all)
    
    # plot results
    active_log = pd.read_csv(f'{args.save_dir}/active_log.csv', header=None).values
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(active_log[:, 0], active_log[:, 1], color=color, marker='o')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Heterosedastic loss of testing data', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    color = 'tab:blue'
    ax2 = ax1.twinx()
    ax2.plot(active_log[:, 0], active_log[:, 2], color=color, marker='o', label='rmse')
    ax2.plot(active_log[:, 0], active_log[:, 3], color='darkcyan', marker='o', label='mae')
    ax2.set_ylabel('RMSE/MAE of testing data', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.suptitle('Active Learning Log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{args.save_dir}/active_log.png', dpi=300)
    