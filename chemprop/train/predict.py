from typing import List, Tuple, Union

import torch
import torch.nn as nn
from tqdm import trange
import numpy as np

from chemprop.data import MoleculeDataset, StandardScaler


def predict(model: nn.Module,
            data: MoleculeDataset,
            batch_size: int,
            sampling_size: int,
            fp_method: str,
            scaler: StandardScaler = None,
            atomic_unc: bool = False) -> Tuple[Union[List[List[float]], None], ...]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []
    ale_unc = []
    epi_unc = []

    atomic_pred = []
    atomic_ales = []

    aleatoric = model.aleatoric
    # if MC-Dropout
    mc_dropout = model.mc_dropout

    num_iters, iter_step = len(data), batch_size

    for i in range(0, num_iters, iter_step):
        # Prepare batch
        mol_batch = MoleculeDataset(data[i:i + batch_size])
        smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

        # Run model
        batch = smiles_batch
        if not aleatoric and not mc_dropout:
            with torch.no_grad():
                batch_preds = model(batch, features_batch)
            batch_preds = batch_preds.data.cpu().numpy()

            # Inverse scale if regression
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)

            # Collect vectors
            batch_preds = batch_preds.tolist()
            preds.extend(batch_preds)

        elif aleatoric and not mc_dropout:
            with torch.no_grad():
                batch_preds, batch_var, batch_atomic_pred, batch_atomic_ales = model(batch, features_batch)
                if fp_method == 'molecular':
                    batch_var = torch.exp(batch_var)  # log_var in molecular fp_method
            batch_preds = batch_preds.data.cpu().numpy()
            batch_ale_unc = batch_var.data.cpu().numpy()

            # Inverse scale if regression
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)
                batch_ale_unc = scaler.inverse_transform_variance(batch_ale_unc)
            # Collect vectors
            batch_preds = batch_preds.tolist()
            batch_ale_unc = batch_ale_unc.tolist()
            preds.extend(batch_preds)
            ale_unc.extend(batch_ale_unc)

            if atomic_unc:
                batch_atomic_pred = batch_atomic_pred.data.cpu().numpy()
                batch_atomic_ales = batch_atomic_ales.data.cpu().numpy()
                if scaler is not None:
                    batch_atomic_pred = scaler.inverse_transform(batch_atomic_pred)
                    batch_atomic_ales = scaler.inverse_transform_variance(batch_atomic_ales)
                atomic_pred.extend(batch_atomic_pred)  # bs x max_atom_size
                atomic_ales.extend(batch_atomic_ales)    # bs x max_atom_size


        
        elif not aleatoric and mc_dropout:
            with torch.no_grad():
                P_mean = []

                for ss in range(sampling_size):
                    batch_preds = model(batch, features_batch)
                    P_mean.append(batch_preds)

                batch_preds = torch.mean(torch.stack(P_mean), 0)
                batch_epi_unc = torch.var(torch.stack(P_mean), 0)

            batch_preds = batch_preds.data.cpu().numpy()
            batch_epi_unc = batch_epi_unc.data.cpu().numpy()
        
        elif aleatoric and mc_dropout:
            with torch.no_grad():
                P_mean = []
                P_var = []
                for ss in range(sampling_size):
                    batch_preds, batch_var, batch_atomic_pred, batch_atomic_ales = model(batch, features_batch)
                    if fp_method == 'molecular':
                        batch_var = torch.exp(batch_var)  # log_var in molecular fp_method
                    P_mean.append(batch_preds)
                    P_var.append(batch_var)

                batch_preds = torch.mean(torch.stack(P_mean), 0)
                batch_ale_unc = torch.mean(torch.stack(P_var), 0)
                batch_epi_unc = torch.var(torch.stack(P_mean), 0)

            batch_preds = batch_preds.data.cpu().numpy()
            batch_ale_unc = batch_ale_unc.data.cpu().numpy()
            batch_epi_unc = batch_epi_unc.data.cpu().numpy()

            # Inverse scale if regression
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)
                batch_ale_unc = scaler.inverse_transform_variance(batch_ale_unc)
                batch_epi_unc = scaler.inverse_transform_variance(batch_epi_unc)

            # Collect vectors
            batch_preds = batch_preds.tolist()
            batch_ale_unc = batch_ale_unc.tolist()
            batch_epi_unc = batch_epi_unc.tolist()

            preds.extend(batch_preds)
            ale_unc.extend(batch_ale_unc)
            epi_unc.extend(batch_epi_unc)

    if atomic_unc:
        atomic_pred = np.r_[atomic_pred]
        atomic_ales = np.r_[atomic_ales]
        print(f'predict.py | atomic_pred.shape: {atomic_pred.shape}, atomic_ales.shape: {atomic_ales.shape}')
        return preds, ale_unc, None, atomic_pred, atomic_ales

    if not aleatoric and not mc_dropout:
        return preds, None, None, None, None
    elif aleatoric and not mc_dropout:
        return preds, ale_unc, None, None, None
    elif not aleatoric and mc_dropout:
        return preds, None, epi_unc, None, None
    elif aleatoric and mc_dropout:
        return preds, ale_unc, epi_unc, None, None