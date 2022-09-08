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


def make_predictions(args: Namespace, smiles: List[str] = None) -> List[Optional[List[float]]]:
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
    for i in range(len(args.checkpoint_paths)):
        scaler_i, feature_scaler_i = load_scalers(args.checkpoint_paths[i])
        print(f'index: {i}, scaler: {scaler_i.means, scaler_i.stds}')
    train_args = load_args(args.checkpoint_paths[0])

    # Update args with training arguments
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

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
    if args.pred_max_atom_size is not None:
        print(f'predict max heavy atom size: {args.pred_max_atom_size}')
        args.max_atom_size = args.pred_max_atom_size
    else:
        print('args.pred_max_atom_size is None')
    # Predict with each model individually and sum predictions
    if args.dataset_type == 'multiclass':
        sum_preds = np.zeros((len(test_data), args.num_tasks, args.multiclass_num_classes))
        sum_ale_uncs = np.zeros((len(test_data), args.num_tasks, args.multiclass_num_classes))
        sum_epi_uncs = np.zeros((len(test_data), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_preds = np.zeros((len(test_data), args.num_tasks))
        sum_ale_uncs = np.zeros((len(test_data), args.num_tasks))
        sum_epi_uncs = np.zeros((len(test_data), args.num_tasks))

    # Partial results for variance robust calculation.
    all_preds = np.zeros((len(test_data), args.num_tasks, len(args.checkpoint_paths)))

    print(pformat(vars(args)))
    
    print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    for index, checkpoint_path in enumerate(tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths), disable=True)):
        # Load model
        model = load_checkpoint(checkpoint_path, current_args=args, cuda=args.cuda)
        model_preds, ale_uncs, epi_uncs, _, _ = predict(
            model=model,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler,
            sampling_size=args.sampling_size,
            fp_method=args.fp_method
        )
        sum_preds += np.array(model_preds)

        ### For mixed model ###
        if ale_uncs is not None:
            sum_ale_uncs += np.array(ale_uncs)
        if epi_uncs is not None:
            sum_epi_uncs += np.array(epi_uncs)
        if args.estimate_variance:
            all_preds[:, :, index] = model_preds
        ### For mixed model ###

    # Ensemble predictions
    ### For mixed model ###
    if args.estimate_variance:  # not mc_dropout
        # Use ensemble variance to estimate uncertainty. This overwrites existing uncertainty estimates.
        # preds <- mean(preds), ale_uncs <- mean(ale_uncs), epi_uncs <- var(preds)
        avg_preds = sum_preds / len(args.checkpoint_paths)
        avg_preds = avg_preds.tolist()

        avg_ale_uncs = sum_ale_uncs / len(args.checkpoint_paths)
        avg_ale_uncs = avg_ale_uncs.tolist()

        avg_epi_uncs = np.var(all_preds, axis=2)
        avg_epi_uncs = avg_epi_uncs.tolist()
    else:  # mc_dropout
        # Use another method to estimate uncertainty.
        # preds <- mean(preds), ale_uncs <- mean(ale_uncs), epi_uncs <- mean(epi_uncs)
        avg_preds = sum_preds / len(args.checkpoint_paths)
        avg_preds = avg_preds.tolist()

        avg_ale_uncs = sum_ale_uncs / len(args.checkpoint_paths)
        avg_ale_uncs = avg_ale_uncs.tolist()

        avg_epi_uncs = sum_epi_uncs / len(args.checkpoint_paths)
        avg_epi_uncs = avg_epi_uncs.tolist()
    ### For mixed model ###

    # Save predictions
    assert len(test_data) == len(avg_preds)
    assert len(test_data) == len(avg_ale_uncs)
    assert len(test_data) == len(avg_epi_uncs)

    print(f'Saving predictions to {args.preds_path}')

    # Put Nones for invalid smiles
    full_preds = [None] * len(full_data)
    full_ale_uncs = [None] * len(full_data)
    full_epi_uncs = [None] * len(full_data)

    for i, si in enumerate(valid_indices):
        full_preds[si] = avg_preds[i]
        full_ale_uncs[si] = avg_ale_uncs[i]
        full_epi_uncs[si] = avg_epi_uncs[i]
    
    avg_preds = full_preds
    avg_ale_uncs = full_ale_uncs
    avg_epi_uncs = full_epi_uncs
    avg_total_uncs = np.array(avg_ale_uncs) + np.array(avg_epi_uncs)

    test_smiles = full_data.smiles()
    ### For mixed model ###

    # Write predictions
    with open(args.preds_path, 'w') as f:
        writer = csv.writer(f)

        header = []

        if args.use_compound_names:
            header.append('compound_names')

        header.append('smiles')

        if args.dataset_type == 'multiclass':
            for name in args.task_names:
                for i in range(args.multiclass_num_classes):
                    header.append(name + '_class' + str(i))
        else:
            if args.write_true_val:
                header.append('true_'+args.task_names[0])
            header.append('pred_'+args.task_names[0])
            header.extend([tn + "_ale_unc" for tn in args.task_names])
            header.extend([tn + "_epi_unc" for tn in args.task_names])
            header.extend([tn + "_total_unc" for tn in args.task_names])

        writer.writerow(header)

        for i in range(len(avg_preds)):
            row = []

            if args.use_compound_names:
                row.append(compound_names[i])

            row.append(test_smiles[i])

            if args.write_true_val:
                row.append(true_vals[i])

            if avg_preds[i] is not None:
                if args.dataset_type == 'multiclass':
                    for task_probs in avg_preds[i]:
                        row.extend(task_probs)
                else:
                    row.extend(avg_preds[i])
                    row.extend(avg_ale_uncs[i])
                    row.extend(avg_epi_uncs[i])
                    row.extend(avg_total_uncs[i])

            else:
                if args.dataset_type == 'multiclass':
                    row.extend([''] * args.num_tasks * args.multiclass_num_classes)
                else:
                    # Both the prediction, the aleatoric uncertainty and the epistemic uncertainty are None
                    row.extend([''] * 3 * args.num_tasks)

            writer.writerow(row)
          
    return avg_preds
