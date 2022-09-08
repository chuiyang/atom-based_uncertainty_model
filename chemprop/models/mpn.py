from argparse import Namespace
from typing import List, Union

import torch
import torch.nn as nn
import numpy as np

from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function, get_cc_dropout_hyper
from chemprop.models.concrete_dropout import ConcreteDropout

import time

torch.set_printoptions(edgeitems=7)


class MPNEncoder(nn.Module):  # for atomic_vecs_d2, atomic_vecs_final, mol_vecs
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.use_input_features = args.use_input_features
        self.max_atom_size = args.max_atom_size
        self.epistemic = args.epistemic
        self.mc_dropout = self.epistemic == 'mc_dropout'
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm
        self.fp_method = args.fp_method
        self.corr_similarity_function = args.corr_similarity_function
        self.args = args
        

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Concrete Dropout for Bayesian NN
        wd, dd = get_cc_dropout_hyper(args.train_data_size, args.regularization_scale)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim   # self.bond_fdim=145

        # cosine similarity
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-8)
        
        if self.mc_dropout:
            self.W_i = ConcreteDropout(layer=nn.Linear(input_dim, self.hidden_size, bias=self.bias), reg_acc=args.reg_acc, weight_regularizer=wd, dropout_regularizer=dd)
        else:
            self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:  # hidden_size
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        if self.mc_dropout:
            self.W_h = ConcreteDropout(layer=nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias), reg_acc=args.reg_acc, weight_regularizer=wd, dropout_regularizer=dd, depth=self.depth - 1)
            self.W_o = ConcreteDropout(layer=nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size), reg_acc=args.reg_acc, weight_regularizer=wd, dropout_regularizer=dd)
        else:
            self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)
            self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)
    
    # zero-padding tensor
    def padding(self, mol_vector):  # chui: set max atom number per molecule to 10
        num_atoms_index = mol_vector.shape[0]
        num_features_index = mol_vector.shape[1]
        padding_tensor = torch.zeros((self.max_atom_size, num_features_index))  # default 10x300
        padding_tensor[:num_atoms_index, :] = mol_vector
        return padding_tensor

    def get_cov_index(self, atom_num):
        b = []
        [b.extend(range(i * self.max_atom_size, i * self.max_atom_size + atom_num)) for i in range(atom_num)]
        return b

    def get_sign(self, val):
        val[val >= 0] = 1
        val[val < 0] = -1
        return val

    def cov_func_padding(self, mol_vector):  # check input Kxy but only output variance
        num_atoms = mol_vector.size(0)
        first = mol_vector.repeat(mol_vector.size(0), 1)
        second = mol_vector.unsqueeze(1).repeat(1, mol_vector.size(0), 1).view(-1, mol_vector.size(1))
        output_tensor = torch.cat((first, second), dim=1)
        mol_dim = mol_vector.size(1)

        # inner product
        if self.corr_similarity_function == 'cos':
            # val = torch.sum(output_tensor[:, :mol_dim]*output_tensor[:, mol_dim:], axis=1)  # inner product
            # cos = val / torch.sqrt(torch.sum(torch.pow(output_tensor[:, :mol_dim], 2), axis=1)) / torch.sqrt(torch.sum(torch.pow(output_tensor[:, mol_dim:], 2), axis=1))
            cos = self.cosine_similarity(output_tensor[:, :mol_dim], output_tensor[:, mol_dim:])
            absolute_tensor = cos.view(-1, 1)

        # RBF kernel
        elif self.corr_similarity_function == 'rbf':
            absolute_tensor = torch.exp(-torch.sum(output_tensor[:, :mol_dim]-output_tensor[:, mol_dim:], axis=1)**2/300).view(-1, 1)
            val = torch.sum(output_tensor[:, :mol_dim]*output_tensor[:, mol_dim:], axis=1)  # inner product
            sign = self.get_sign(val).view(-1, 1)
            absolute_tensor = absolute_tensor*sign

        elif self.corr_similarity_function == 'pearson':
            pearson = self.cosine_similarity(output_tensor[:, :mol_dim] - output_tensor[:, :mol_dim].mean(dim=1, keepdim=True), output_tensor[:, mol_dim:] - output_tensor[:, mol_dim:].mean(dim=1, keepdim=True))
            absolute_tensor = pearson.view(-1, 1)

        else:
            raise ValueError(f'atomic fingerprint similarity function {self.corr_similarity_function} is not supported.')

        padding_cov_tensor = torch.zeros((self.max_atom_size*self.max_atom_size, absolute_tensor.size(1)))
        place_index = self.get_cov_index(num_atoms)
        if self.args.cuda:
            padding_cov_tensor = padding_cov_tensor.cuda()
        padding_cov_tensor[place_index, :] = absolute_tensor[:, :]

        return padding_cov_tensor

    def forward(self,
                mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            if self.args.cuda:
                features_batch = features_batch.cuda()

            if self.features_only:
                return features_batch

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, conv_bt = mol_graph.get_components()  # wei fix

        if self.atom_messages:
            a2a = mol_graph.get_a2a()

        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb, conv_bt = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda(), conv_bt.cuda()  # wei fix

            if self.atom_messages:
                a2a = a2a.cuda()

        # Input
        if self.atom_messages:  # false
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:  # False
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden
       
            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        ############ without relu ##############
        atom_hiddens = self.W_o(a_input)  # num_atoms x hidden  # norelu!!, self.act_func(self.W_o(a_input))
        ############ without relu ##############
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = []
        cov_vecs = []
        asize_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                if self.fp_method != 'atomic':
                    if self.aggregation == 'mean':
                        mol_vec_molar = mol_vec.sum(dim=0) / a_size
                    elif self.aggregation == 'sum':
                        mol_vec_molar = mol_vec.sum(dim=0)
                    elif self.aggregation == 'norm':
                        mol_vec_molar = mol_vec.sum(dim=0) / self.aggregation_norm
                    
                # mol_vec_molar is molecular fingerprint
                if self.fp_method == 'molecular':          
                    mol_vecs.append(mol_vec_molar)

                else: # atomic or hybrid method
                    if self.fp_method == 'atomic':
                        pass
                    elif self.fp_method == 'hybrid_dim0':
                        mol_vec = torch.cat((mol_vec, mol_vec_molar.view(1, -1)), dim=0)
                        assert mol_vec.shape[0] == (a_size + 1)
                    else: # 'hybrid_dim1'
                        assert self.fp_method == 'hybrid_dim1'
                        mol_vec_molar = mol_vec_molar.repeat(a_size, 1)
                        assert mol_vec_molar.shape == mol_vec.shape
                        mol_vec = torch.cat((mol_vec, mol_vec_molar), dim=1)
                    # padding
                    new_mol_vec = self.padding(mol_vec)
                    new_cov_vec = self.cov_func_padding(mol_vec)
                
                    mol_vecs.append(new_mol_vec)
                    cov_vecs.append(new_cov_vec)
                    if self.args.intensive_property:
                        asize_vecs.append(torch.tensor(a_size))
                    
        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, num_atoms, hidden_size)
        if self.args.cuda:
            mol_vecs = mol_vecs.cuda()
        if self.fp_method in ['atomic', 'hybrid_dim0', 'hybrid_dim1']:
            cov_vecs = torch.stack(cov_vecs, dim=0)
            if self.args.cuda:
                cov_vecs = cov_vecs.cuda()
            
            if self.args.intensive_property:
                asize_vecs = torch.stack(asize_vecs, dim=0).unsqueeze(1)
                if self.args.cuda:
                    asize_vecs = asize_vecs.cuda()


        # if self.use_input_features:
        #     features_batch = features_batch.to(mol_vecs)
        #     if len(features_batch.shape) == 1:
        #         features_batch = features_batch.view([1,features_batch.shape[0]])
        #     mol_vecs = torch.cat([mol_vecs, features_batch], dim=1)  # (num_molecules, num_atoms,  hidden_size)

        # num_molecules x num_atoms x hidden , num_molecules x num_atoms^2 x 1 , num_molecules x 1
        return mol_vecs, cov_vecs, asize_vecs  
    

class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        super(MPN, self).__init__()  # equals to nn.Module.__init__()
        self.features_only = args.features_only
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim  # self.bond_fdim=145 where bond_fdim=17, atom_fdim=128
        self.graph_input = graph_input
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

        if self.features_only:
            return

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, num_atoms, hidden_size) containing the encoding of each molecule.
        """
        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch, self.args)

        output = self.encoder.forward(batch, features_batch)
        return output



