from argparse import Namespace
import argparse
from typing import List, Tuple, Union

from rdkit import Chem
import torch

# Atom feature sizes
MAX_ATOMIC_NUM = 100

ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values ; + 1 for mass 
# + 1 for whether the atom is in ring ; + 7 for whether the atom is in [3, 4, 5, 6, 7, 8] member rings or not
# + 2 for num_atom_in_ring & num_bond_in_ring
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 1 + 1 + 7 # + 2


# + 7 for whether the bond is in [3, 4, 5, 6, 7, 8] member rings
# -4 for remove bondtype
BOND_FDIM = 14 + 7 - 4

# Memoization
SMILES_TO_GRAPH = {}


def clear_cache():
    """Clears featurization cache."""
    global SMILES_TO_GRAPH
    SMILES_TO_GRAPH = {}


def get_atom_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of atom features.

    :param: Arguments.
    """
    return ATOM_FDIM


def get_bond_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of bond features.

    :param: Arguments.
    """
    return BOND_FDIM


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, mol: Chem.rdchem.Mol = None, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           [1 if atom.IsInRing() else 0] + \
           atom_in_member_rings(atom) + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            # bt == Chem.rdchem.BondType.SINGLE,
            # bt == Chem.rdchem.BondType.DOUBLE,
            # bt == Chem.rdchem.BondType.TRIPLE,
            # bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
        fbond += bond_in_member_rings(bond)
    return fbond


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, smiles: str, args: Namespace):
        """
        Computes the graph structure and featurization of a molecule.

        :param smiles: A smiles string.
        :param args: Arguments.
        """
        self.smiles = smiles
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.conv_bt = []  # wei fix, if non-single bond == 1 else == 0

        # Convert identifiers to molecule
        mol = Chem.MolFromSmiles(smiles)

        # fake the number of "atoms" if we are collapsing substructures
        self.n_atoms = mol.GetNumAtoms()
        
        if args.covariance_matrix_pred:
            assert args.covariance_matrix_save_path is not None
            print(f"featurization.py162 | writing data at: {f'{args.covariance_matrix_save_path}'}")
            log_file = open(f'{args.covariance_matrix_save_path}', 'w')
            log_file.write(f'{self.smiles}\n')
            log_file.close()

        # Get atom features
        for i, atom in enumerate(mol.GetAtoms()):
            self.f_atoms.append(atom_features(atom, mol=mol))

            if args.covariance_matrix_pred:
                assert args.covariance_matrix_save_path is not None
                log_file = open(f'{args.covariance_matrix_save_path}', 'a')
                log_file.write('%s, %s, %s\n' % (atom.GetSmarts(), atom.GetSmarts(), atom.GetTotalNumHs()))
                log_file.close()

        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]   # ??? while len(mol.GetAtoms) == mol.GetNumAtoms

        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = bond_features(bond)

                if args.atom_messages:  # false
                    self.f_bonds.append(f_bond)
                    self.f_bonds.append(f_bond)
                else:
                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # wei fix, convolution bond type
                bt = bond.GetBondType() 
                if bt != Chem.rdchem.BondType.SINGLE:
                    self.conv_bt.append(1)  # a1 --> a2
                    self.conv_bt.append(1)  # a2 --> a1
                else:
                    self.conv_bt.append(0)  # a1 --> a2
                    self.conv_bt.append(0)  # a2 --> a1

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2


class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraph], args: Namespace):
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)

        self.atom_fdim = get_atom_fdim(args)
        self.bond_fdim = get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        conv_bt = [0]  # wei fix, convolution bond type, if non-single bond == 1 else == 0, star from 0 as padding

        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])
                conv_bt.append(mol_graph.conv_bt[b])  # wei fix

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b)) # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages
        self.conv_bt = torch.LongTensor(conv_bt)  # wei fix

    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope, self.conv_bt  # wei fix

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a


def mol2graph(smiles_batch: List[str],
              args: Namespace) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    mol_graphs = []
    for smiles in smiles_batch:
        if smiles in SMILES_TO_GRAPH:
            mol_graph = SMILES_TO_GRAPH[smiles]
        else:
            mol_graph = MolGraph(smiles, args)
            if not args.no_cache:  # no_cache == False
                SMILES_TO_GRAPH[smiles] = mol_graph
        mol_graphs.append(mol_graph)
    
    return BatchMolGraph(mol_graphs, args)


def is_zwitterion(mol: Chem.rdchem.Mol):
    """
    To identify whether the molecule is zwitterion or not 
    """
    zwitterion = 0
    for atom in mol.GetAtoms():
       if atom.GetFormalCharge() !=0:
           zwitterion = 1
           break
       
    return [zwitterion]


def atom_in_member_rings(atom: Chem.rdchem.Atom):
    """
    Show each atom of the molecule is involved in 3, 4, 5, 6, 7, 8 member rings
    """
    vector = [0]*7  # If value is not in the [3, 4, 5, 6, 7, 8], then the final element in the vector is 1.
    for index, member_ring in enumerate(range(3, 9)):
        if atom.IsInRingSize(member_ring):
            vector[index] = 1
    
    if vector == [0]*7:
        vector[-1] = 1
    
    return vector


def bond_in_member_rings(bond: Chem.rdchem.Bond):
    """
    Show each bond of the molecule is involved in 3, 4, 5, 6, 7, 8 member rings
    """
    vector = [0]*7  # If value is not in the [3, 4, 5, 6, 7, 8], then the final element in the vector is 1.
    for index, member_ring in enumerate(range(3, 9)):
        if bond.IsInRingSize(member_ring):
            vector[index] = 1
    
    if vector == [0]*7:
        vector[-1] = 1
    
    return vector



def with_message_passing(mol: Chem.rdchem.Mol):
    """
    Molecule with heavy atom(s)=1 or 2 is not processed message passing action. (Only for D-MPNN)
    """
    m_passing_vector = [0, 0, 0]  # m_passing_vector = [hv=1, hv=2, hv>2]
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 1:
        m_passing_vector[0] = 1
    elif num_atoms == 2:
        m_passing_vector[1] = 1
    else:
        m_passing_vector[2] = 1

    return m_passing_vector


def num_atom_in_ring(mol: Chem.rdchem.Mol):
    """
    Check the number of the atoms that are in the ring,, count vector.
    """
    count = 0
    for atom in mol.GetAtoms():
        if atom.IsInRing():
            count += 1

    return [count]


def num_bond_in_ring(mol: Chem.rdchem.Mol):
    """
    Check the number of the bonds that are in the ring, count vector.
    """
    count = 0
    n_atoms = mol.GetNumAtoms()
    for a1 in range(n_atoms):
        for a2 in range(a1 + 1, n_atoms):
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond is None:
                continue
            elif bond.IsInRing():
                count += 1

    return [count]

#mol = Chem.MolFromSmiles('C1=CC=CC=C1')
#print(num_atom_in_ring(mol))
#print(num_bond_in_ring(mol))
#print(with_message_passing(mol))

'''
def is_hetroatomic_cyclic(mol: Chem.rdchem.Mol):
    """
    Show whether molecule with ring structure is C-only cyclic or not
    """
    vector = [0]*3  # [N involve, O involve, C-only]
    member_ring = 0
    C_in_ring_count = 0

    for atom in mol.GetAtoms():
        if atom.IsInRing():
            member_ring += 1
            if atom.GetAtomicNum() == 7:
                vector[0] = 1
            if atom.GetAtomicNum() == 8:
                vector[1] = 1
            if atom.GetAtomicNum() == 6:
                C_in_ring_count += 1
    
    if member_ring == C_in_ring_count and member_ring != 0:
        vector[2] = 1
    
    return vector
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--atom_messages', action='store_true', default=False)    
    parser.add_argument('--no_cache', action='store_true', default=False)
    args = parser.parse_args()
    graph = MolGraph('C#CC1=C(C#C)CCC1', args)  # C#CC1=C(C#C)CCC1
    batch_graph = BatchMolGraph([graph], args)
