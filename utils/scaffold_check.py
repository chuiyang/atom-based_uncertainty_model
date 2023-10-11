import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


train_data = pd.read_csv('./saved_models/qm9_130k_scaf/qm9_130k_pear_scale_2l_21e_scaf_lr/fold_0/train_smiles.csv').values[:, 0]

scaf_set = set()
for train_smiles in train_data:

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(train_smiles))
    scaf_set.add(scaffold)

print(f'len(scaf_set): {len(scaf_set)}')



output = []


ccsd = pd.read_csv('./data/new_heavy_atom/new_binary/heavy_atom_9.csv').values
print(f'ccsd.shape: {ccsd.shape}')

for ccsd_smiles, ccsd_hf in zip(ccsd[:, 0], ccsd[:, 1]):
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(ccsd_smiles))
    if scaffold not in scaf_set:
        output.append([ccsd_smiles, ccsd_hf])

print(f'len(output): {len(output)}')


ccsd = pd.read_csv('./data/new_heavy_atom/new_binary/heavy_atom_10.csv').values
print(f'ccsd.shape: {ccsd.shape}')
for ccsd_smiles, ccsd_hf in zip(ccsd[:, 0], ccsd[:, 1]):
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(ccsd_smiles))
    if scaffold not in scaf_set:
        output.append([ccsd_smiles, ccsd_hf])

output = np.array(output)

print(f'output.shape: {output.shape}')


pd.DataFrame(output, columns=['smiles', 'Hf']).to_csv('./saved_models/qm9_130k_scaf/qm9_130k_pear_scale_2l_21e_scaf_lr/fold_0/ccsd_hf_scaffoldNotInTrain.csv', index=False)