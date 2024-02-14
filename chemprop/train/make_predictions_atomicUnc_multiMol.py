
import os
from typing import List
import logging
import numpy as np
import torch
from tqdm import tqdm
from pprint import pformat
from argparse import Namespace

from .predict import predict
from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data, get_data_from_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers
from chemprop.atom_plot.molecule_drawer import MoleculeDrawer


def draw_and_save_molecule(i, smiles_i, mol_unc_i, atomic_unc_i, unc_t, args, svg=False, logger=None):
    smiles = smiles_i
    mol_unc = float(mol_unc_i)
    atom_uncs = [round(a, 2) for a in atomic_unc_i.astype(float)]
    try:
        pic_data = MoleculeDrawer.draw_molecule_with_atom_notes(smiles=smiles, mol_note=mol_unc, atom_notes=atom_uncs, unc_type=unc_t, svg=svg)
    except:
        if logger:
            logger.error(f'Cannot draw molecule {i}: {smiles_i}')
        else:
            print(f'[Error] Cannot draw molecule {i}: {smiles_i}')
        return False
    if svg:
        with open(os.path.join(args.unc_type_png_path, f'{i}_{unc_t}.svg'), 'w') as f:
            f.write(pic_data)
    else:
        with open(os.path.join(args.unc_type_png_path, f'{i}_{unc_t}.png'), 'wb') as f:
            f.write(pic_data)
    return True
    
def make_predictions_atomicUnc_multiMol(args: Namespace, smiles: List[str] = None, logger: logging.Logger = None) -> None:
    """
    Makes predictions. If smiles is provided, makes predictions on smiles. Otherwise makes predictions on args.test_data.

    :param args: Arguments.
    :param smiles: Smiles to make predictions on.
    :return: None.
    """
    high_resolution = args.high_resolution
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    logger.info('Loading training args')
    scaler, features_scaler = load_scalers(args.checkpoint_paths[0])
    train_args = load_args(args.checkpoint_paths[0])

    # Update args with training arguments
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    
    args.atomic_unc = True
    logger.info(pformat(vars(args)))
    logger.info('Loading data')
    if smiles is not None:
        test_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False)
    else:
        if args.write_true_val:
            test_data, true_vals = get_data(path=args.test_path, args=args, use_compound_names=args.use_compound_names, skip_invalid_smiles=False)
        else:
            test_data = get_data(path=args.test_path, args=args, use_compound_names=args.use_compound_names, skip_invalid_smiles=False)

    logger.info('Validating SMILES')
    valid_indices = [i for i in range(len(test_data)) if test_data[i].mol is not None]
    full_data = test_data
    test_data = MoleculeDataset([test_data[i] for i in valid_indices])

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    if args.use_compound_names:
        compound_names = test_data.compound_names()
    logger.info(f'Test size = {len(test_data):,}')

    # Normalize features
    if train_args.features_scaling:
        test_data.normalize_features(features_scaler)

    # max atom size check
    args.max_atom_size = 0
    logger.info(f'Checking testing data max HeavyAtom size')
    for test_mol in test_data.mols():
        if test_mol.GetNumHeavyAtoms() > args.max_atom_size:
            args.max_atom_size = args.pred_max_atom_size = test_mol.GetNumHeavyAtoms()
    logger.info(f'Max heavy atom size = {args.max_atom_size}')
    if args.covariance_matrix_pred:
        args.scaler_stds = scaler.stds
        logger.info(f'covariance matrix pred: scaling factor = {args.scaler_stds}')
    # Predict with each model individually and sum predictions
    all_preds = np.zeros((len(test_data), len(args.checkpoint_paths)))
    all_ale_uncs = np.zeros((len(test_data), len(args.checkpoint_paths)))
    all_atomic_preds = np.zeros((len(test_data), args.max_atom_size, len(args.checkpoint_paths)))
    all_atomic_ales = np.zeros((len(test_data), args.max_atom_size, len(args.checkpoint_paths)))
    
    logger.info(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    for index, checkpoint_path in enumerate(tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths), disable=True)):
        # Load model
        model = load_checkpoint(checkpoint_path, current_args=args, cuda=args.cuda, logger=logger)
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
        all_atomic_preds[:, :, index] = atomic_preds  # num_mols x atom_preds x models
        all_atomic_ales[:, :, index] = atomic_uncs  # num_mols x atom_preds x models

    # Ensemble predictions
    assert args.estimate_variance is not None
    avg_preds = (np.sum(all_preds, axis=1) / len(args.checkpoint_paths))[:, np.newaxis]
    avg_ale_uncs = (np.sum(all_ale_uncs, axis=1) / len(args.checkpoint_paths))[:, np.newaxis]
    avg_epi_uncs = np.var(all_preds, axis=1)[:, np.newaxis]
    avg_total_uncs = np.array(avg_ale_uncs) + np.array(avg_epi_uncs)
    avg_test_atomic_preds = np.mean(all_atomic_preds, axis=2)
    avg_test_atomic_ales = np.sum(all_atomic_ales, axis=2) / len(args.checkpoint_paths)
    avg_test_atomic_epis = np.var(all_atomic_preds, axis=2)
    avg_test_atomic_total = avg_test_atomic_ales + avg_test_atomic_epis  # mol x max_atom_size
    # avg_test_atomic_max_ales = np.max(avg_test_atomic_ales, axis=1)[:, np.newaxis].tolist()  # take max ale_unc of atoms in a molecule
    # avg_test_atomic_max_epis = np.max(avg_test_atomic_epis, axis=1)[:, np.newaxis].tolist()
    # avg_test_atomic_max_total = np.max(avg_test_atomic_total, axis=1)[:, np.newaxis].tolist()
    # del avg_test_atomic_ales, avg_test_atomic_epis, avg_test_atomic_total

    # Save predictions
    assert len(test_data) == len(avg_preds)
    assert len(test_data) == len(avg_ale_uncs)
    assert len(test_data) == len(avg_epi_uncs)
    test_smiles = full_data.smiles()


    args.png_path = os.path.join(args.draw_mols_dir)
    unc_type = ['epi', 'ale', 'pred']

    logger.info(f'make image directory: {args.png_path}')
    os.makedirs(args.png_path, exist_ok=True)

    for unc_t in unc_type:
        args.unc_type_png_path = os.path.join(args.png_path, unc_t)
        os.makedirs(args.unc_type_png_path) if not os.path.isdir(args.unc_type_png_path) else None
        if unc_t == 'ale':
            mol_unc = avg_ale_uncs
            atomic_unc = avg_test_atomic_ales
        elif unc_t == 'epi':
            mol_unc = avg_epi_uncs
            atomic_unc = avg_test_atomic_epis
        elif unc_t == 'total':
            mol_unc = avg_total_uncs
            atomic_unc = avg_test_atomic_total
        elif unc_t == 'pred':
            mol_unc = avg_preds
            atomic_unc = avg_test_atomic_preds
        for i, (smiles_i, mol_unc_i, atomic_unc_i) in enumerate(zip(test_smiles, mol_unc, atomic_unc)):
            draw_and_save_molecule(i, smiles_i, mol_unc_i, atomic_unc_i, unc_t, args, svg=high_resolution, logger=logger)
    return avg_preds
