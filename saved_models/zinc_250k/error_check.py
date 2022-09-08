"""

check the powerfulness of ensemble
the fraction of data improved after ensmeble

"""
import pandas as pd
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt


preds_array = []
# aleas_array = []
for model in range(30):
    file_before = pd.read_csv(f'./zinc_250k_pear_scale_2l_bs50_30e/each_pred/zinc_250k_test_pear_scale_2l_bs50_30e_{model}.csv')
    preds_array.append(file_before['preds_logP'].values)
    # aleas_array.append(np.sqrt(file_before['Hf_ale_unc'].values))

preds_array = np.array(preds_array)
# aleas_array = np.array(aleas_array)
# print(f'aleas_array: {aleas_array[:, :5]}')
print(f'preds_array.shape: {preds_array.shape}')
assert preds_array.shape[0] == 30


file_after = pd.read_csv('./zinc_250k_pear_scale_2l_bs50_30e/fold_0/zinc_250k_test_pear_scale_2l_bs50_30e.csv')

smiles = file_after['smiles'].values
preds_ens = file_after['preds_logP'].values
true = file_after['true_logP'].values
error = abs(preds_ens-true)
ale_unc = np.sqrt(file_after['logP_ale_unc'].values)
epi_unc = file_after['logP_epi_unc'].values

for model_i, pred_array in enumerate(preds_array):
    ens_good = 0
    total = 0
    for (pred_i, pred_ens, true_i) in zip(pred_array, preds_ens, true):
        total += 1
        if abs(pred_i - true_i) > abs(pred_ens - true_i):
            ens_good += 1
    
    print(f'model {model_i}: {ens_good} / {total} ens better: {round(ens_good/total, 2)}')

