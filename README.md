# Atom-Based Uncertainty Model for Molecular Property Prediction

The atom-based uncertainty model is modified from the architecture of the molecule-based uncertainty model: https://github.com/gscalia/chemprop/tree/uncertainty


The atom-based uncertainty quantification method provides an extra layer of explainability to both aleatoric and epistemic uncertainties, i.e., one can analyze individual atomic uncertainty values to diagnose the chemical component that introduces the uncertainty in the prediction.

Note:
Currently support regression task only.


## Training
* You can train **atom-based uncertainty model** by running:
```
python train.py \
--data_path <training_data_path> \
--save_dir <save_path> \
--max_atom_size 9 \
--dataset_type regression \
--epochs 150 \
--no_features_scaling \
--seed 20 \
--aleatoric \
--metric heteroscedastic \
--fp_method atomic \
--corr_similarity_function pearson \
--y_scaling \
--batch_size 50 \
--save_smiles_splits \
--max_lr 5e-4 \
--ensemble_size 15
```
`<training_data_path>` is the CSV file with columns name at the first row
e.g.
smiles,logP
CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F,3.112
CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1,2.432

| Attempt | #1    |
| :---:   | :---: |
| Seconds | 301   | 

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |


`<save_path>` is the path to save the checkpoints.
`--max_atom_size` is to specify the largest size of molecule in the training data.
e.g. the maximum number of atoms in a molecule is 9.

* You can train **molecule-based uncertainty model** by running:
```
python train.py \
--data_path <training_data_path> \
--save_dir <save_path> \
--dataset_type regression \
--epochs 150 \
--no_features_scaling \
--seed 20 \
--aleatoric \
--metric heteroscedastic \
--fp_method molecular \
--aggregation sum \
--y_scaling \
--batch_size 50 \
--save_smiles_splits \
--ensemble_size 30 \
--max_lr 5e-4 
```
`<training_data_path>` and `<save_path>` is the same as the above, and no need to specify the largest size of molecule in the training data when training a molecule-based uncertainty model 

## Evaluating

