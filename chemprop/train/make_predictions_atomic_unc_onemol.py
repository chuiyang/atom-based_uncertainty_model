from argparse import Namespace
import csv
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from pprint import pformat

from .predict import predict
from .evaluate import evaluate_predictions
from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data, get_data_from_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers, get_metric_func

from rdkit import Chem

def make_predictions_atomic_unc_onemol(args: Namespace, smiles: List[str] = None) -> List[Optional[List[float]]]:
    """
    Makes predictions. If smiles is provided, makes predictions on smiles. Otherwise makes predictions on args.test_data.

    :param args: Arguments.
    :param smiles: Smiles to make predictions on.
    :return: A list of lists of target predictions.
    """
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    print('Loading training args')
    scaler, features_scaler = load_scalers(args.checkpoint_paths[0])
    train_args = load_args(args.checkpoint_paths[0])

    # Update args with training arguments
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    
    args.atomic_unc = True
    print(pformat(vars(args)))
    print('Loading data')
    if smiles is not None:
        test_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False)
    else:
        if args.write_true_val:
            test_data, true_vals = get_data(path=args.test_path, args=args, use_compound_names=args.use_compound_names, skip_invalid_smiles=False)
        else:
            test_data = get_data(path=args.test_path, args=args, use_compound_names=args.use_compound_names, skip_invalid_smiles=False)

    print('Validating SMILES')
    valid_indices = [i for i in range(len(test_data)) if test_data[i].mol is not None]
    full_data = test_data
    test_data = MoleculeDataset([test_data[i] for i in valid_indices])

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    if args.use_compound_names:
        compound_names = test_data.compound_names()
    print(f'Test size = {len(test_data):,}')

    # Normalize features
    if train_args.features_scaling:
        test_data.normalize_features(features_scaler)
    # max atom size check
    args.max_atom_size = args.pred_max_atom_size = test_data.mols()[0].GetNumHeavyAtoms()
    print(f'args.max_atom_size = {args.max_atom_size}')
    if args.covariance_matrix_pred:
        args.scaler_stds = scaler.stds
        print(f'covariance matrix pred: scaling factor = {args.scaler_stds}')
    # Predict with each model individually and sum predictions
    all_preds = np.zeros((len(test_data), len(args.checkpoint_paths)))
    all_ale_uncs = np.zeros((len(test_data), len(args.checkpoint_paths)))
    all_atomic_preds = np.zeros((len(test_data), args.max_atom_size, len(args.checkpoint_paths)))
    all_atomic_ales = np.zeros((len(test_data), args.max_atom_size, len(args.checkpoint_paths)))
    
    print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    for index, checkpoint_path in enumerate(tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths), disable=True)):
        # Load model
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
        all_atomic_preds[:, :, index] = atomic_preds  # num_mols x atom_preds x models
        all_atomic_ales[:, :, index] = atomic_uncs  

    # Ensemble predictions
    assert args.estimate_variance is not None
    avg_preds = (np.sum(all_preds, axis=1) / len(args.checkpoint_paths))[:, np.newaxis].tolist()
    avg_ale_uncs = (np.sum(all_ale_uncs, axis=1) / len(args.checkpoint_paths))[:, np.newaxis].tolist()
    avg_epi_uncs = np.var(all_preds, axis=1)[:, np.newaxis].tolist()
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

    print(f'Saving predictions to {args.preds_path}')

    # Put Nones for invalid smiles
    full_preds = [None] * len(full_data)
    full_ale_uncs = [None] * len(full_data)
    full_epi_uncs = [None] * len(full_data)
    full_atomic_max_ale = [None] * len(full_data)
    full_atomic_max_epi = [None] * len(full_data)
    full_atomic_max_total = [None] * len(full_data)

    for i, si in enumerate(valid_indices):
        full_preds[si] = avg_preds[i]
        full_ale_uncs[si] = avg_ale_uncs[i]
        full_epi_uncs[si] = avg_epi_uncs[i]
        # full_atomic_max_ale[si] = avg_test_atomic_max_ales[i]
        # full_atomic_max_epi[si] = avg_test_atomic_max_epis[i]
        # full_atomic_max_total[si] = avg_test_atomic_max_total[i]
    
    avg_preds = full_preds
    avg_ale_uncs = full_ale_uncs
    avg_epi_uncs = full_epi_uncs
    avg_total_uncs = np.array(avg_ale_uncs) + np.array(avg_epi_uncs)
    avg_atomic_max_ale = full_atomic_max_ale
    avg_atomic_max_epi = full_atomic_max_epi
    avg_atomic_max_total = full_atomic_max_total

    test_smiles = full_data.smiles()
    print(f'writing predictions number: {len(avg_preds)}')
    # Write predictions
    print(f'writing file in {args.preds_path}')
    with open(args.preds_path, 'w') as f:
        writer = csv.writer(f)

        for i in range(len(avg_preds)):
            smiles_row = []
            mol = Chem.MolFromSmiles(test_smiles[i])
            count = 0
            for atom in mol.GetAtoms():
                print(atom.GetSmarts())
                smiles_row.append(atom.GetSmarts())
                count += 1
            print(f'atom number: {count}')
            print(smiles_row)
            writer.writerow(['smiles', test_smiles[i]]+ smiles_row)
            writer.writerow(['preds', avg_preds[i][0]] + list(avg_test_atomic_preds[i]))
            writer.writerow(['ale', avg_ale_uncs[i][0]] + list(avg_test_atomic_ales[i]))
            writer.writerow(['epi', avg_epi_uncs[i][0]] + list(avg_test_atomic_epis[i]))
            writer.writerow(['total', avg_total_uncs[i][0]] + list(avg_test_atomic_total[i]))
            
            print([test_smiles[i]] + list(test_smiles[i]))
            print([avg_preds[i]] + list(avg_test_atomic_preds[i]))
            print([avg_ale_uncs[i]] + list(avg_test_atomic_ales[i]))
            print([avg_epi_uncs[i]] + list(avg_test_atomic_epis[i]))
            print([avg_total_uncs[i]] + list(avg_test_atomic_total[i]))
            print(np.sum(avg_test_atomic_preds[i]))

          
    return avg_preds
