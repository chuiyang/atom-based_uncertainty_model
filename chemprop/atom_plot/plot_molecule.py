from rdkit import Chem
import rdkit.Chem.Draw as Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
import pandas as pd
import argparse

"""
use rdkit to draw molecule. Higher uncertainty will results in red marks.
"""

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx()+1)
    return mol

def highlight_substructure(mol, mol_unc, atomic_unc):
    hl_atoms = []
    hl_bonds = []
    avg_unc = mol_unc / mol.GetNumAtoms()
    print(f'mol_unc: {mol_unc:.3f}, mol.GetNumAtoms(): {mol.GetNumAtoms()}, avg_unc: {avg_unc:.3f}')
    
    for a, a_unc in zip(mol.GetAtoms(), atomic_unc):
        print(f'atom index: {a.GetIdx()}')
        if a_unc > avg_unc:
            hl_atoms.append(a.GetIdx())
    for b in mol.GetBonds():
        b1, b2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        print(f'bonds index: {b.GetIdx()}')
        if (b1 in hl_atoms) and (b2 in hl_atoms):
            hl_bonds.append(b.GetIdx())
    return hl_atoms, hl_bonds


def draw_with_index(mol_file, args):
    smiles = mol_file['smiles'].iloc[0]
    mol_unc = float(mol_file[args.unc_type].iloc[0])
    atomic_unc = mol_file[args.unc_type].iloc[1:].values.astype(float)

    mol = Chem.MolFromSmiles(smiles)
    print(Chem.MolToSmiles(mol))
    mol = mol_with_atom_index(mol)

    # mol = Chem.MolFromSmiles('C1CC23C4CC2(C1CC23OC4CC(C12)C34)CN13')
    # print(mol)

    # test
    # mol.GetSubstructMatches(Chem.MolFromSmarts('C(=O)O'))
    hl_atoms, hl_bonds = highlight_substructure(mol, mol_unc, atomic_unc)
    # a = Draw.MolToImage(mol, highlightAtoms=hl_atoms, highlightBonds=hl_bonds)  #, size=max_size, kekulize=kekulize, options=options, canvas=canvas, **kwargs)

    d2d = rdMolDraw2D.MolDraw2DSVG(750,700)
    d2d.DrawMolecule(mol, highlightAtoms=hl_atoms, highlightBonds=hl_bonds)
    d2d.FinishDrawing()

    png_data = d2d.GetDrawingText()

    # # save png to file
    # with open(f'{args.png_name}_{args.unc_type}.png', 'wb') as png_file:
    #     png_file.write(png_data)
    import cairosvg
    import tempfile

    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp.write(png_data.encode())
        tmp.flush()
        cairosvg.svg2png(url=tmp.name, write_to=f'{args.png_name}_{args.unc_type}.png')

    # a.save(f'{args.png_name}_{args.unc_type}.png')

if __name__ == '__main__':
    # draw_with_index('C1CCC1')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='./../../saved_models/backup_ccsd_tran_goodmodel/ccsd_9k_cos_scale_2l_15e_trans/acyclic_pred.csv')
    parser.add_argument('--unc_type', type=str, default='total')
    parser.add_argument('--png_name', type=str, default='test')
    args = parser.parse_args()
    
    mol_file = pd.read_csv(args.test_path, header=None)
    print(mol_file.head)
    mol_file = mol_file.T
    mol_file.columns = mol_file.iloc[0]
    mol_file = mol_file.drop(0)
    print('--------------')
    print(mol_file)
    draw_with_index(mol_file, args)