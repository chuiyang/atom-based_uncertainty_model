from rdkit.Chem import AllChem
from rdkit import Chem

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


def has_atom(smile):
    atomSymbol = 'N'
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