# Atom-Based Uncertainty Model for Molecular Property Prediction

![image](https://github.com/chuiyang/atom-based_uncertainty_model/blob/main/TOC.jpeg)

The atom-based uncertainty model is modified from the architecture of the molecule-based uncertainty model: https://github.com/gscalia/chemprop/tree/uncertainty


The atom-based uncertainty quantification method provides an extra layer of explainability to both aleatoric and epistemic uncertainties, i.e., one can analyze individual atomic uncertainty values to diagnose the chemical component that introduces the uncertainty in the prediction.

![image](https://github.com/chuiyang/atom-based_uncertainty_model/blob/main/TableOfContentsGraph.001.jpeg)

Note:
Currently support regression task only.


## Training
### Train **atom-based uncertainty model** by running:
```bash
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
* `<training_data_path>` is the CSV file with columns name at the first row

e.g.
| smiles  | logP  |
| :---:   | :---: |
| COCC(=O)N(C)CC(=O)NCC1(Nc2nccn3nnnc23)CC1 | -1.315   | 
| CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1 | 2.432   | 
| ... | ... |

* `<save_path>` is the path to save the checkpoints.
* `--max_atom_size` is to specify the largest size of molecule in the training data.
e.g. the maximum number of atoms in a molecule is 9.

### Train **molecule-based uncertainty model** by running:
```bash
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
* `<training_data_path>` and `<save_path>` is the same as the above, and there is no need to specify the largest size of molecule in the training data when training a molecule-based uncertainty model 

## Evaluating
Currently, you can predict:
1. A CSV file with molecules in SMILES format --outputs--> molecular (property prediction/aleatoric/epistemic/total uncertainty)
2. A CSV file with molecules in SMILES format --outputs--> svg. images of molecules with atomic (contribution/aleatoric/epistemic) labeled near the atom.

### 1. Run:
```
python predict.py \
--test_path <eval_path> \
--checkpoint_dir <model_path> \
--preds_path <pred_path> \
--estimate_variance 
```
* `<eval_path>` is the CSV file path to evaluate
* `<model_path>` is the checkpoint path where the model is saved. It should be the `<save_path>` when you train the model.
* `<pred_path>` is the path to save the output file after predicting the `<eval_path>`

### 2. Run:
```
python predict_atomicunc_multiMol.py \
--test_path <eval_path> \
--checkpoint_dir <model_path> \
--preds_path <pred_path> \
--estimate_variance 
```
* by running predict_atomicunc_multiMol.py, a folder named by the CSV file name of `<eval_path>` will be created. The svg. images of molecules will be saved in the folder.
e.g.
`<eval_path>` == './molecule/test_data.csv'
/test_data
   // pred
   // ale
   // epi
svg. images of (pred/ale/epi) will be saved in the directory

## Post-hoc recalibration

To fine-tune the variance layer in either atom- or molecule-based uncertainty model, run train_multimodel.py and add `--transfer_learning_freeze_GCNN`.

In the following, the ensemble model before post-hoc calibration is named as "ens_model" and the ensemble model after post-hoc calibration is named as "post-hoc_ens_model".

### Train the pos-hoc recalibration on ens_model by running:
```bash
python train_multimodel.py \
--data_path <training_data_path> \
--separate_val_path <val_data_path> \
--separate_test_path <test_data_path> \
--checkpoint_dir <ens_model_checkpoint_directory> \
--save_dir <post-hoc_ens_model_checkpoint_directory> \
--transfer_learning_freeze_GCNN \
--fp_method molecular \
--init_lr 1e-6 \
--max_lr 1e-5 \
--final_lr 8e-7 \
--warmup_epochs 4 \
--dataset_type regression \
--epochs 150 \
--no_features_scaling \
--seed 20 \
--aleatoric \
--metric heteroscedastic \
--aggregation sum \
--ensemble_size 30 \
--y_scaling \
```
* `<training_data_path>`, `<val_data_path>`, and `<test_data_path>` are the CSV file paths of training/validation/testing data that used in ens_model.
* `<ens_model_checkpoint_directory>` is the path to the saved checkpoint of the ens_model.
* `<post-hoc_ens_model_checkpoint_directory>` is the path to save the checkpoints of post-hoc_ens_model.
* `--transfer_learning_freeze_GCNN` is to freeze the weights that do not belongs in **variance layer**.

