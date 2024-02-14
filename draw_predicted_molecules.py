"""Loads a trained model checkpoint and makes predictions on a dataset."""
import os
import logging

from chemprop.parsing import parse_predict_args
from chemprop.train import make_predictions_atomicUnc_multiMol


def setup_logger(name, log_file, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setLevel(level)

    formatter = logging.Formatter(format)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

if __name__ == '__main__':
    args = parse_predict_args()

    log_file_path = os.path.join(args.draw_mols_dir, 'draw_molecules.log')
    logger = setup_logger('', log_file_path)
    make_predictions_atomicUnc_multiMol(args, smiles=None, logger=logger)

