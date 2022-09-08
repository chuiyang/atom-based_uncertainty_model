from argparse import Namespace
import csv
from tkinter.font import Font
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


# draw mol
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps
from rdkit.Chem import Draw
import os
import rdkit.Geometry.rdGeometry as Geometry
from rdkit.Chem import AllChem
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import svgutils.transform as sg
import re

atomSymbol = 'N'


def unsave_atomUnc_large(mol, atomic_unc):
    for a, a_unc in zip(mol.GetAtoms(), atomic_unc):
        if (a.GetSymbol() == 'N') and (a_unc == max(atomic_unc)):
            return False
    return True


def mol_with_atom_index(mol, atomic_unc=None):

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx()+1)
    if atomic_unc is not None:
        for atom, a_unc in zip(mol.GetAtoms(), atomic_unc):
            atom.SetProp('atomNote', f'{a_unc:.1f}')
        #     atom.SetProp('_displayLabel', '')
        # for bond in mol.GetBonds():
        #     bond.SetProp('displayLabel', '')
        #     bond.SetProp('displayLabelW', '')
    return mol


def highlight_substructure(mol, mol_unc, atomic_unc):
    hl_atoms = []
    hl_bonds = []
    avg_unc = mol_unc / mol.GetNumAtoms()   
    save = True 
    for a, a_unc in zip(mol.GetAtoms(), atomic_unc):
        if a_unc > avg_unc:
            hl_atoms.append(a.GetIdx())
            # if (a.GetSymbol() == 'O'):  ## if highlight O then do not save
                # save = False
            # elif (a.GetSymbol() == 'N') and (a_unc < 1):
                # save = False
        # elif a.GetSymbol() == 'N':  ## if N is not highlight then do not save
            # save = False
        # if (a_unc == max(atomic_unc)) and (a.GetSymbol() != 'N'): ## if max atomic unc is not N
            # save = False
    for b in mol.GetBonds():
        b1, b2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if (b1 in hl_atoms) and (b2 in hl_atoms):
            hl_bonds.append(b.GetIdx())
    return hl_atoms, hl_bonds, save


def titlePos(mol):
    min_x, min_y = 0, 0
    AllChem.EmbedMolecule(mol)
    mh_conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        pos = mh_conf.GetAtomPosition(atom.GetIdx())    
        min_x = pos.x if min_x > pos.x else min_x
        min_y = pos.y if min_y > pos.y else min_y
        print(f'atom.GetIdx(): {atom.GetIdx()}, {pos.x}, {pos.y}')
    return min_x, min_y


def draw_with_index(i, smiles_i, mol_unc_i, atomic_unc_i, unc_t, args):
    smiles = smiles_i
    mol_unc = float(mol_unc_i)
    atomic_unc = atomic_unc_i.astype(float)

    ### classify by number of Nitrogens
    numN = 0
    for atom in Chem.MolFromSmiles(smiles).GetAtoms():
        if atom.GetSymbol() == atomSymbol:
            numN += 1
    numN = 3 if numN > 3 else numN
    num_path = os.path.join(args.png_path, unc_t, f'{numN}{atomSymbol}')
    ### classify by number of Nitrogens


    molInd = Chem.MolFromSmiles(smiles)
    print(f'{Chem.MolToSmiles(molInd), smiles}')
    print(f'Mol_Unc: {mol_unc:.3f}, {unc_t}')
    print(f'mol.GetNumAtoms(): {molInd.GetNumAtoms()}, avg_unc: {[round(a, 3) for a in atomic_unc]}')
    
    if unc_t == 'pred':
        d2d = rdMolDraw2D.MolDraw2DSVG(520, 550)  # (600, 670)
        d2d.DrawMolecule(molInd)
        mix = False

    elif mol_unc < 1.2:
        print(unc_t, mol_unc)
        d2d = rdMolDraw2D.MolDraw2DSVG(520, 550)
        # save = unsave_atomUnc_large(molInd, atomic_unc)
        d2d.DrawMolecule(molInd)
        mix = False
    else:
        d2d = rdMolDraw2D.MolDraw2DSVG(520, 550)  # MolDraw2DCairo
        # d2d = rdMolDraw2D.MolDraw2DCairo(520, 550) # MolDraw2DCairo
        # hl_atoms, hl_bonds, save = highlight_substructure(molInd, mol_unc, atomic_unc)
        mix = True
        ###
        colors = [(1, 0.2, 0.2), (1, 1, 1), (1, 0.2, 0.2)] # pink
        cmap = LinearSegmentedColormap.from_list('self_define', colors, N=100)
        SimilarityMaps.GetSimilarityMapFromWeights(molInd, list(atomic_unc), colorMap=cmap, contourLines=2, draw2d=d2d, alpha=0.5, sigma=0.25) #0.34
        ###

    d2d.SetFontSize(44)
    height = 3.1  # 3.5
    # if unc_t == 'pred':
    #     d2d.DrawString(f'Prediction: {mol_unc:.2f}', Geometry.Point2D(0, height))
    # elif unc_t == 'ale':
    #     d2d.DrawString(f'Aleatoric: {mol_unc:.2f}', Geometry.Point2D(0, height))
    # elif unc_t == 'epi':
    #     d2d.DrawString(f'Epistemic: {mol_unc:.2f}', Geometry.Point2D(0, height))
    # else:
    #     d2d.DrawString(f'Total: {mol_unc:.2f}', Geometry.Point2D(0, height)) #-2.6        
    d2d.FinishDrawing()

    png_data = d2d.GetDrawingText()
    # open(os.path.join(num_path, f'{i}_{unc_t}.svg'),'wb+').write(png_data) 
    with open(os.path.join(num_path, f'{i}_{unc_t}.svg'), 'w') as f:
        f.write(png_data)

    if mix:
        d2d2 = rdMolDraw2D.MolDraw2DSVG(520, 550)

        opts = d2d2.drawOptions()
        opts.clearBackground=False

        # d2d2.DrawMolecule(mol_with_atom_index(Chem.MolFromSmiles(smiles), atomic_unc))  # with atomic notes (uncertainty)
        ###down
        d2d2.DrawMolecule(mol_with_atom_index(Chem.MolFromSmiles(smiles)))  # without atomic notes (uncertainty)
        d2d2.SetFontSize(44)

        ###
        if unc_t == 'pred':
            d2d2.DrawString(f'Prediction: {mol_unc:.2f}', Geometry.Point2D(0, height))
        elif unc_t == 'ale':
            d2d2.DrawString(f'Aleatoric: {mol_unc:.2f}', Geometry.Point2D(0, height))
        elif unc_t == 'epi':
            d2d2.DrawString(f'Epistemic: {mol_unc:.2f}', Geometry.Point2D(0, height))
        else:
            d2d2.DrawString(f'Total: {mol_unc:.2f}', Geometry.Point2D(0, height)) #-2.6     
        ###
        d2d2.FinishDrawing()

        png_data2 = d2d2.GetDrawingText()
        # open(os.path.join(num_path, f'{i}_{unc_t}_mol.svg'),'wb+').write(png_data2)
        with open(os.path.join(num_path, f'{i}_{unc_t}_mol.svg'), 'w') as f:
            f.write(png_data2)

        # blend image
        # background = Image.open(os.path.join(num_path, f'{i}_{unc_t}.svg'))
        # overlay = Image.open(os.path.join(num_path, f'{i}_{unc_t}_mol.svg'))
        background = sg.fromfile(os.path.join(num_path, f'{i}_{unc_t}.svg'))
        overlay = sg.fromfile(os.path.join(num_path, f'{i}_{unc_t}_mol.svg'))
        def convert_to_pixels(measurement):
            value = float(re.search(r'[0-9\.]+', measurement).group())
            if measurement.endswith("px"):
                return value
            elif measurement.endswith("mm"):
                return value * 3.7795275591
            else:
                # unit not supported
                return value

        width = convert_to_pixels(background.get_size()[0])
        height = convert_to_pixels(background.get_size()[1])
        logo_width = convert_to_pixels(overlay.get_size()[0])
        logo_height = convert_to_pixels(overlay.get_size()[1])
        root = overlay.getroot()
        background.append([root])
        background.save(os.path.join(num_path, f'{i}_{unc_t}_all.svg'))


        # background = background.convert("RGBA")
        # overlay = overlay.convert("RGBA")

        # # transparant 
        # datas = overlay.getdata()
        # newOverlay = []
        # for items in datas:
        #     if items[0] >= 200 and items[1] >= 200 and items[2] >= 200:
        #         newOverlay.append((255, 255, 255, 0))
        #     else:
        #         newOverlay.append(items)
        # overlay.putdata(newOverlay)

        # background.paste(overlay, (0, 0), overlay)
        # background.save(os.path.join(num_path, f'{i}_{unc_t}_all.svg'),"PNG") 

    return True

def has_atom(smile):
    for atom in Chem.MolFromSmiles(smile).GetAtoms():
        if atom.GetSymbol() == atomSymbol:
            return True
    return False

def atomsize(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol.GetNumHeavyAtoms() < 9:
        return True
    else:
        return False
    
def make_predictions_atomicUnc_multiMol(args: Namespace, smiles: List[str] = None, draw: bool = True) -> None:
    """
    Makes predictions. If smiles is provided, makes predictions on smiles. Otherwise makes predictions on args.test_data.

    :param args: Arguments.
    :param smiles: Smiles to make predictions on.
    :return: None.
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
    args.max_atom_size = args.pred_max_atom_size
    print(f'args.max_atom_size = {args.max_atom_size}')
    print(f'checking testing data max HeavyAtom size')
    for test_mol in test_data.mols():
        if test_mol.GetNumHeavyAtoms() > args.max_atom_size:
            args.max_atom_size = args.pred_max_atom_size = test_mol.GetNumHeavyAtoms()
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

    print(f'Saving predictions to {args.preds_path}')

    test_smiles = full_data.smiles()

    save_smiles = []
    mol_uncs = []
    indexs = []

    # draw
    if draw:
        args.png_path = os.path.join(args.checkpoint_dir, 'fold_0', args.test_path.split('/')[-1].replace('.csv', ''))
        unc_type = ['epi', 'ale', 'pred']  # , , 'epi', 'total', 'pred'

        print(f'make png directory: {args.png_path}')
        os.makedirs(args.png_path, exist_ok=True)
        

        for unc_t in unc_type:
            os.makedirs(os.path.join(args.png_path, unc_t), exist_ok=True)

            for n in range(4):
                os.makedirs(os.path.join(args.png_path, unc_t, f'{n}{atomSymbol}'), exist_ok=True)

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
                # if (mol_unc_i < 1) and (has_atom(smiles_i)):
                # if (mol_unc_i > 3) and (has_atom(smiles_i)): # and (has_atom(smiles_i)):  #  and (atomsize(smiles_i))
                # if (has_atom(smiles_i)):
                if True:
                    save = draw_with_index(i, smiles_i, mol_unc_i, atomic_unc_i, unc_t, args)
                    if save:
                        save_smiles.append(smiles_i)
                        mol_uncs.extend(mol_unc_i)
                        indexs.append(i)
                else:
                    continue

    # pd.DataFrame(np.array([save_smiles, mol_uncs]).T, columns=['smiles', 'Hf']).to_csv(os.path.join(args.checkpoint_dir, 'fold_0', 'epi_demo_withoutN_17.csv'), index=False)
    # pd.DataFrame(np.array([save_smiles, mol_uncs, indexs]).T, columns=['smiles', 'Hf', 'index']).to_csv(os.path.join(args.checkpoint_dir, 'fold_0', 'epi_demo_withoutN_17_withIndex.csv'), index=False)
          
    return avg_preds
