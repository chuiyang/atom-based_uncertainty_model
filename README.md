# Atom-Based Uncertainty Model

![image](https://github.com/chuiyang/atom-based_uncertainty_model/blob/main/TOC.jpeg)

The atom-based uncertainty quantification method provides an extra layer of explainability to both aleatoric and epistemic uncertainties, i.e., one can analyze individual atomic uncertainty values to diagnose the chemical component that introduces the uncertainty in the prediction.

Detailed content is available at ChemRxiv:

[Explainable Uncertainty Quantifications for Deep Learning-Based Molecular Property Prediction](https://doi.org/10.26434/chemrxiv-2022-qt49t)



The atom-based uncertainty model is modified from the architecture of the molecule-based uncertainty model: https://github.com/gscalia/chemprop/tree/uncertainty

Note:
Currently only **regression tasks** are supported.
This repository is still under development. (19.12.2022)

## Computational Cost
The computational cost depends on **the size of the training set** and **the number of epochs the machine runs**.

We give the user a little idea of how long it takes to train the model.

The times shown below are for training an atom-based uncertainty model.

(If you want 5 models to form an ensemble model, 5 times the time needs to be considered if you do not perform parallel processing during training.)

For **Delaney**, the size of dataset is 1128 molecules. We split train:val:test to 8:1:1. We set the 150 epochs with early stopping if no improvement in 50 epochs.
|    Epochs it runs    | Time |
| ------------- | ------------- |
| 60  | 2 mins 33 secs |
| 65  | 2 mins 37 secs |
| 69  | 2 mins 55 secs |
| 88  | 3 mins 44 secs |
| 101 | 6 mins 08 sces |

For **QM9**, the size of dataset is 134k molecules. We split train:val:test to 8:1:1. We set the 150 epochs with early stopping if no improvement in 15 epochs.
|    Epochs it runs    | Time |
| ------------- | ------------- |
| 36  | 141 mins  |
| 70  | 258 mins  |
| 114 | 337 mins  |

All timings were performed on 4 cores of a 2.0GHz AMD EPYC Rome 64-core processor machine.

## Training
### Train **atom-based uncertainty model** by running:
```bash
python train.py \
--data_path <training_data_path> \
--save_dir <save_path> \
--max_atom_size 9 \
--fp_method atomic \
--corr_similarity_function pearson \
--dataset_type regression \
--epochs 150 \
--no_features_scaling \
--seed 20 \
--aleatoric \
--metric heteroscedastic \
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
* `--fp_method` should be specified as `atomic` for atomic predictive distributions.

### Train **molecule-based uncertainty model** by running:
```bash
python train.py \
--data_path <training_data_path> \
--save_dir <save_path> \
--dataset_type regression \
--fp_method molecular \
--epochs 150 \
--no_features_scaling \
--seed 20 \
--aleatoric \
--metric heteroscedastic \
--aggregation sum \
--y_scaling \
--batch_size 50 \
--save_smiles_splits \
--ensemble_size 30 \
--max_lr 5e-4 
```
* `<training_data_path>` and `<save_path>` is the same as the above, and there is no need to specify the largest size of molecule in the training data when training a molecule-based uncertainty model 
* `--fp_method` should be specified as `molecular` for only generating molecular predictive distribution.

## Evaluating
Currently, you can predict:
1. A CSV file with molecules in SMILES format --outputs--> molecular (property prediction/aleatoric/epistemic/total uncertainty)
2. A CSV file with molecules in SMILES format --outputs--> SVG images of molecules with atomic (contribution/aleatoric/epistemic) labeled near the atoms.

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
--estimate_variance \
--pred_max_atom_size 1
```
* by running predict_atomicunc_multiMol.py, a folder named by the CSV file name of `<eval_path>` will be created. The svg. images of molecules will be saved in the folder.

e.g.
`<eval_path>` == './molecule/test_data.csv'

A folder named `test_data` contains `pred`, `ale`, and `epi` folders.

Three SVG images are generated per molecule, including property prediction, aleatoric uncertainty, and epistemic uncertainty.

These SVG images will be classified into the folders they belong to. (e.g. aleatoric uncertainty with atomic aleatoric uncertainty image is in `ale` folder)

![image](https://github.com/chuiyang/atom-based_uncertainty_model/blob/main/image.jpeg)

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
--y_scaling 
```
* `<training_data_path>`, `<val_data_path>`, and `<test_data_path>` are the CSV file paths of training/validation/testing data that used in ens_model.
* `<ens_model_checkpoint_directory>` is the path to the saved checkpoint of the ens_model.
* `<post-hoc_ens_model_checkpoint_directory>` is the path to save the checkpoints of post-hoc_ens_model.
* `--transfer_learning_freeze_GCNN` is to freeze the weights that do not belongs in **variance layer**.

