# Atom-Based Uncertainty Model

![image](https://github.com/chuiyang/atom-based_uncertainty_model/blob/main/images/TOC.jpeg)

The atom-based uncertainty quantification method provides an extra layer of explainability to both aleatoric and epistemic uncertainties, i.e., one can analyze individual atomic uncertainty values to diagnose the chemical component that introduces the uncertainty in the prediction.

Detailed content is available at Journal of Cheminformatics:

[Explainable Uncertainty Quantifications for Deep Learning-Based Molecular Property Prediction](https://doi.org/10.1186/s13321-023-00682-3)



The atom-based uncertainty model is modified from the architecture of a [molecule-based uncertainty model](https://github.com/gscalia/chemprop/tree/uncertainty) by Scalia G, Grambow CA, Pernici B et al (2020)

> **Note**:
Currently only **regression tasks** are supported.
This repository is still under development. (14.02.2024) \
I'll try to deliver this repo biweekly, hopefully~ :)

## Table of Contents
- [Train](#train)
- [Evaluate](#evaluate)
  - [Molecular Property Prediction](#1-molecular-property-prediction)
  - [Draw Molecular Images](#2-draw-molecular-images)
- [Post-hoc recalibration](#post-hoc-recalibration)


## Train
### Train **atom-based uncertainty model** by running:
```bash
python train.py \
--data_path <training_data_path> \
--save_dir <save_dir_path> \
--dataset_type regression \
--max_atom_size 9 \
--aleatoric \
--metric heteroscedastic \
```
[Below is optional for atom-based uncertainty model]
```bash
--fp_method atomic \
--corr_similarity_function pearson \
--epochs 150 \
--no_features_scaling \
--seed 20 \
--y_scaling \
--batch_size 50 \
--save_smiles_splits \
--max_lr 1e-3 \
--ensemble_size 1
```
* `<training_data_path>` is the CSV file with columns name at the first row

e.g.
| smiles  | [property name]  |
| :---:   | :---: |
| c1ccccc1 | -1.31   | 
| CCCO | 2.43   | 
| ... | ... |

* `<save_dir_path>` is the path to save the checkpoints. e.g., 👉 ./result/folder1
* `--max_atom_size` is to specify the largest size of molecule in the training data.
e.g. the maximum number of atoms in a molecule is 9.
* `--fp_method` should be specified as `atomic` for atomic predictive distributions. (default: atomic)

### Train **molecule-based uncertainty model** by running:
```bash
python train.py \
--data_path <training_data_path> \
--save_dir <save_dir_path> \
--dataset_type regression \
--fp_method molecular \
--aleatoric \
--metric heteroscedastic \
```
[Below is optional for molecule-based uncertainty model]
```bash
--epochs 150 \
--no_features_scaling \
--seed 20 \
--aggregation sum \
--y_scaling \
--batch_size 50 \
--save_smiles_splits \
--ensemble_size 30 \
--max_lr 5e-4 
```
* `<training_data_path>` and `<save_dir_path>` is the same as the above, and there is no need to specify the largest size of molecule in the training data when training a molecule-based uncertainty model 
* `--fp_method` should be specified as `molecular` for only generating molecular predictive distribution.

## Evaluate
Currently, you can predict:
1. **Input**: A CSV file with molecules in SMILES format \
**Output**: Molecular (property/aleatoric uncertainty/epistemic uncertainty) predictions. (CSV file)

2. **Input**: A CSV file with molecules in SMILES format \
**Output**: PNG/SVG images of molecules with atomic (contribution/aleatoric/epistemic) labeled near the atoms. (a folder with PNG/SVG files)

### 1. Molecular Property Prediction 
Run:
```bash
python predict.py \
--test_path <test_path> \
--checkpoint_dir <model_dir_path> \
--preds_path <pred_path> \
--estimate_variance 
```
* `<test_path>` is the CSV file path to evaluate. e.g., 👉 ./data/test_data.csv
* `<model_dir_path>` is the checkpoint directory path where the model is saved. It should be same as the `<save_dir_path>` when you train the model. e.g., 👉 ./result/folder1
* `<pred_path>` is the CSV file path to save the output file after predicting the `<test_path>`e.g., 👉 ./data/test_data_pred.csv
* Note: if the maximum heavy atom size in the test_path is larger then training data in checkpoint_dir, add `--pred_max_atom_size <size>` tag (atom-based uncertainty model only, to be fixed).

### 2. Draw Molecular Images
with atomic information \
Run:
```bash
python draw_predicted_molecules.py \
--test_path <test_path> \
--checkpoint_dir <model_dir_path> \
--draw_mols_dir <pred_dir_path> \
--high_resolution
```
* `<test_path>` is the CSV file path to evaluate. e.g., 👉 ./data/test_data.csv
* `<model_dir_path>` is the checkpoint directory path where the model is saved. It should be same as the `<save_dir_path>` when you train the model. e.g., 👉 ./result/folder1
* `<pred_dir_path>` is the directory path where the images will be saved. 👉 ./molecule/test_data_image_folder 
> A folder named `test_data_image_folder` contains `pred`, `ale`, and `epi` folders. \
Three PNG/SVG images will be generated per molecule, including property prediction, aleatoric uncertainty, and epistemic uncertainty. \
These PNG/SVG images will be classified into the folders they belong to. (e.g. aleatoric uncertainty with atomic aleatoric uncertainty image is in `ale` folder)
* `--high_resolution` add this tag will generate images with svg format. Else, images with png format.

![image](https://github.com/chuiyang/atom-based_uncertainty_model/blob/main/images/draw_predicted_molecule_images.png)

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

## Computational Cost
The computational cost depends on **the size of the training set** and **the number of epochs the machine runs**.<br />We give the user an idea of how long it takes to train the model.<br />The times shown below are for training an atom-based uncertainty model.<br />(If you want 5 models to form an ensemble model, 5 times the time needs to be considered if you do not perform parallel processing during training.)

For **Delaney**, the size of dataset is 1128 molecules. We split train:val:test into 8:1:1, set the 150 epochs, and stop early if there is no improvement in 50 epochs.
|    Epochs it ran    | Time |
| ------------- | ------------- |
| 60  | 2 mins 33 secs |
| 65  | 2 mins 37 secs |
| 69  | 2 mins 55 secs |
| 88  | 3 mins 44 secs |
| 101 | 6 mins 08 sces |

For **QM9**, the size of dataset is 134k molecules. We split train:val:test into 8:1:1, set the 150 epochs, and stop early if there is no improvement in 15 epochs.
|    Epochs it ran    | Time |
| ------------- | ------------- |
| 36  | 141 mins  |
| 70  | 258 mins  |
| 90  | 330 mins  |
| 94  | 352 mins  |
| 114 | 411 mins  |
| 116 | 424 mins  |
| 130 | 472 mins  |

All timings above were performed on 4 cores of a 2.0GHz AMD EPYC Rome 64-core processor machine.


For **Delaney**, the size of dataset is 1128 molecules. We split train:val:test into 8:1:1, set the 150 epochs, and stop early if there is no improvement in 50 epochs.
|    Epochs it ran    | Time |
| ------------- | ------------- |
| 74  | 2 mins 47 secs |
| 81  | 3 mins 07 secs |
| 78  | 3 mins 01 secs |
| 79  | 3 mins 04 secs |
| 104 | 3 mins 57 sces |

For **QM9**, the size of dataset is 134k molecules. We split train:val:test into 8:1:1, set the 150 epochs, and stop early if there is no improvement in 15 epochs.
|    Epochs it ran    | Time |
| ------------- | ------------- |
| 31  | 117 mins  |
| 36  | 134 mins  |
| 61  | 211 mins  |
| 85  | 286 mins  |
| 108 | 366 mins  |

All timings above were performed on 8 cores of a 2.0GHz AMD EPYC Rome 64-core processor machine.