"""model covariance matrix"""
from argparse import Namespace

import torch.nn as nn
import torch

from .mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights, get_cc_dropout_hyper
from chemprop.models.concrete_dropout import ConcreteDropout, RegularizationAccumulator


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, multiclass: bool, aleatoric: bool, epistemic: str, 
                 fp_method:str, atomic_unc: bool, twoUnitOutput:bool, intensive_property:bool,
                 covariance_matrix_pred:bool, covariance_matrix_save_path: str = None):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)

        self.aleatoric = aleatoric
        self.epistemic = epistemic
        self.mc_dropout = self.epistemic == 'mc_dropout' 
        self.fp_method = fp_method
        self.atomic_unc = atomic_unc
        self.twoUnitOutput = twoUnitOutput
        self.intensive_property = intensive_property
        self.covariance_matrix_pred = covariance_matrix_pred
        self.covariance_matrix_save_path = covariance_matrix_save_path

    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.max_atom_size = args.max_atom_size
        self.ffn_hidden_size = args.ffn_hidden_size
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size  # default = 300
            if args.use_input_features:  # default = false
                first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        wd, dd = get_cc_dropout_hyper(args.train_data_size, args.regularization_scale)

        # Create FFN layers ['molecular', 'atomic']
        if self.fp_method == 'molecular':
            if args.ffn_num_layers == 1:
                ffn = [
                    dropout,
                ]
                last_linear_dim = first_linear_dim
            else:
                ffn = [
                    dropout,
                    ConcreteDropout(layer=nn.Linear(first_linear_dim, args.ffn_hidden_size),
                                    reg_acc=args.reg_acc, weight_regularizer=wd,
                                    dropout_regularizer=dd) if self.mc_dropout else
                    nn.Linear(first_linear_dim, args.ffn_hidden_size)
                ]
                for _ in range(args.ffn_num_layers - 2):
                    ffn.extend([
                        activation,
                        dropout,
                        ConcreteDropout(layer=nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                                        reg_acc=args.reg_acc, weight_regularizer=wd,
                                        dropout_regularizer=dd) if self.mc_dropout else
                        nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size)
                    ])
                ffn.extend([
                    activation,
                    dropout,

                ])
                last_linear_dim = args.ffn_hidden_size
            # Create FFN model
            self._ffn = nn.Sequential(*ffn)
            if self.aleatoric:
                self.output_layer = nn.Linear(last_linear_dim, args.output_size)
                self.logvar_layer = nn.Linear(last_linear_dim, args.output_size)
            else:
                self.output_layer = nn.Linear(last_linear_dim, args.output_size)

        else:  # atomic, hybrid_dim0, hybrid_dim1
            if args.fp_method == 'hybrid_dim1':
                first_linear_dim = first_linear_dim*2
            if args.ffn_num_layers == 1:
                ffn = [
                    TimeDistributed_wrapper(dropout),
                ]
                last_linear_dim = first_linear_dim

            else:   # ffn_num_layers default=2
                ffn_mean_func = [
                    TimeDistributed_wrapper(dropout),
                    TimeDistributed_wrapper(
                        ConcreteDropout(layer=nn.Linear(first_linear_dim, args.ffn_hidden_size),
                                        reg_acc=args.reg_acc, weight_regularizer=wd,
                                        dropout_regularizer=dd) if self.mc_dropout else
                        nn.Linear(first_linear_dim, args.ffn_hidden_size, bias=False)
                        )
                ]

                for _ in range(args.ffn_num_layers - 2):
                    ffn_mean_func.extend([
                        TimeDistributed_wrapper(activation),
                        TimeDistributed_wrapper(dropout),
                        TimeDistributed_wrapper(
                            ConcreteDropout(layer=nn.Linear(first_linear_dim, args.ffn_hidden_size),
                                            reg_acc=args.reg_acc, weight_regularizer=wd,
                                            dropout_regularizer=dd) if self.mc_dropout else
                            nn.Linear(first_linear_dim, args.ffn_hidden_size, bias=False)
                            ),
                    ])
                ffn_mean_func.extend([
                    TimeDistributed_wrapper(activation),
                    TimeDistributed_wrapper(dropout),
                    # TimeDistributed_wrapper(nn.Linear(args.ffn_hidden_size, args.output_size+1, bias=False)),

                ])
                last_linear_dim = args.ffn_hidden_size

            # Create FFN model
            self._ffn = nn.Sequential(*ffn_mean_func)
            if self.aleatoric:
                if self.twoUnitOutput:
                    self.output_layer = TimeDistributed_wrapper(nn.Linear(last_linear_dim, args.output_size+1, bias=False))
                else:
                    self.output_layer = TimeDistributed_wrapper(nn.Linear(last_linear_dim, args.output_size, bias=False))
                    self.std_layer = TimeDistributed_wrapper(nn.Linear(last_linear_dim, args.output_size, bias=False))
            else:
                self.output_layer = TimeDistributed_wrapper(nn.Linear(last_linear_dim, args.output_size, bias=False))

    def split_y_var(self, y_var_vector):
        output = torch.unsqueeze(y_var_vector[:, :, 0], 2)
        var = torch.unsqueeze(y_var_vector[:, :, 1], 2)
        return output, var

    def call_scaler(self, args):
        """ pred covariance matrix need scaler """
        self.scaling_factor = args.scaler_stds
        print(f'scaling factor = {self.scaling_factor}')

    def compute_var(self, std, corr):  # std: 3, 10, 1 corr: 3, 100, 1
        max_atom_size = self.max_atom_size
        first = std.repeat(1, max_atom_size, 1)
        second = std.unsqueeze(2).repeat(1, 1, max_atom_size, 1).view(std.size(0), -1, 1)
        output_tensor = torch.cat((first, second), dim=2)
        std_ij = (output_tensor[:, :, 0] * output_tensor[:, :, 1]).unsqueeze(2)  # + 1e-10
        covar = corr*std_ij

        if self.covariance_matrix_pred:
            assert self.covariance_matrix_save_path is not None
            log_file = open(f'{self.covariance_matrix_save_path}', 'a')
            log_file.write('covariance matrix\n')
            for i in range(max_atom_size):
                atom = covar.view(1, max_atom_size**2)[0, i*max_atom_size:(i+1)*max_atom_size] * self.scaling_factor**2
                out = atom.cpu().data.numpy().tolist()
                out = [round(o, 6) for o in out]
                [log_file.write(f'{i:.5f}, ') for i in out]
                log_file.write('\n')
                # log_file.write('%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n'.format(out))
            log_file.write('correlation coefficient\n')
            for i in range(max_atom_size):
                atom = corr.view(1, max_atom_size**2)[0, i*max_atom_size:(i+1)*max_atom_size]
                out = atom.cpu().data.numpy().tolist()
                out = [round(o, 6) for o in out]
                # print(out)
                [log_file.write(f'{i:.5f}, ') for i in out]
                log_file.write('\n')
                # log_file.write('%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n'.format(out))
            log_file.close()

        if not self.atomic_unc:
            covar = torch.sum(covar, dim=1)
            return covar, None
        else: # active learning, pred atomic unc
            main_var = torch.diagonal(covar.view(-1, max_atom_size, max_atom_size), dim1=1, dim2=2)  # main_var: bs x max_atom_size(var)
            covar = torch.sum(covar, dim=1)
            return covar, main_var

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        if self.fp_method == 'molecular':
            hid_vec, _, _ = self.encoder(*input)
            _output = self._ffn(hid_vec)
            # print(f'\n_output[:5]:{_output[:5]}')

            if self.aleatoric:
                output = self.output_layer(_output)
                logvar = self.logvar_layer(_output)

                # Gaussian uncertainty only for regression, directly returning in this case
                return output, logvar, None, None
            else:
                output = self.output_layer(_output)

                # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
                if self.classification and not self.training:
                    raise ValueError('Classification is not yet supported.')
                if self.multiclass:
                    raise ValueError('multi-Classification is not yet supported.')
                return output

        else:  # atomic and hybrid_dim0, hybrid_dim1
            if self.aleatoric:
                hid_vec, corr_vec, asize_vec = self.encoder(*input)

                hid_vec = self._ffn(hid_vec)
                if self.twoUnitOutput:
                    _output_vector = self.output_layer(hid_vec)
                    _output_atomic, _output_std = self.split_y_var(_output_vector)
                else:
                    _output_atomic = self.output_layer(hid_vec)
                    _output_std = self.std_layer(hid_vec)  # _output_std : bs x atom num x 1
                
                # write atomic prediction
                if self.covariance_matrix_pred:
                    assert self.covariance_matrix_save_path is not None
                    log_file = open(f'{self.covariance_matrix_save_path}', 'a')
                    log_file.write('atomic prediction\n')
                    a_pred = (_output_atomic*(self.scaling_factor)).squeeze().cpu().data.numpy().tolist()
                    a_pred = [round(o, 6) for o in a_pred]
                    [log_file.write(f'{i}, ') for i in a_pred]
                    log_file.write('\n')
                    log_file.close()
                    print([f'{i}, ' for i in a_pred], end='')
                output = torch.sum(_output_atomic, 1)
                output_std = torch.abs(_output_std) + 1e-10
                # Gaussian uncertainty only for regression, directly returning in this case
                if self.atomic_unc:
                    var, main_var = self.compute_var(output_std, corr_vec)
                    atomic_pred = _output_atomic.squeeze(dim=-1)
                    if self.intensive_property:
                        output = output / asize_vec
                        atomic_pred = atomic_pred / asize_vec
                        var = var / (asize_vec**2)
                        main_var = main_var / (asize_vec**2)
                    return output, var, atomic_pred, main_var  # atomic_pred, main_var shape: bs x max_atom_num
                else:
                    var, _ = self.compute_var(output_std, corr_vec)
                    if self.intensive_property:
                        output = output / asize_vec
                        var = var / (asize_vec**2)
                    return output, var, None, None

            else:  # aleatoric = false
                hid_vec, cov_vec = self.encoder(*input)
                output = self.output_layer(hid_vec)
            
                # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
                if self.classification and not self.training:
                    output = self.sigmoid(output)
                if self.multiclass:
                    output = output.reshape((output.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
                    if not self.training:
                        output = self.multiclass_softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

                return output


def build_model(args: Namespace, scaler=None) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes

    if args.epistemic == 'mc_dropout':
        args.reg_acc = RegularizationAccumulator()

    # for old model without this argument when it was training.
    try:
        args.covariance_matrix_save_path
    except:
        args.covariance_matrix_save_path = None

    model = MoleculeModel(classification=args.dataset_type == 'classification', 
                          multiclass=args.dataset_type == 'multiclass', aleatoric=args.aleatoric, 
                          epistemic=args.epistemic, fp_method=args.fp_method, atomic_unc=args.atomic_unc, 
                          twoUnitOutput=args.twoUnitOutput, intensive_property=args.intensive_property,
                          covariance_matrix_pred=args.covariance_matrix_pred,
                          covariance_matrix_save_path=args.covariance_matrix_save_path)
    if args.covariance_matrix_pred:
        model.call_scaler(args)
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)

    if args.epistemic == 'mc_dropout':
        args.reg_acc.initialize(cuda=args.cuda)

    return model


class LambdaLayer(nn.Module):
    def __init__(self, lambda_function):
        super(LambdaLayer, self).__init__()
        self.lambda_function = lambda_function

    def forward(self, x):
        return self.lambda_function(x)
    

class TimeDistributed_wrapper(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed_wrapper, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y
