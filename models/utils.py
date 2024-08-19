import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import yaml
from models.Loss import Loss_VAE


class Model_Config:
    def __init__(self, model_config_file):
        self.input_dim = model_config_file['input_dim']
        self.num_layers = model_config_file['num_layers']
        self.d_tokens = model_config_file['d_token']
        self.head_num = model_config_file['head_num']
        self.attention_hidden_dim = model_config_file['attention_hidden_dim']
        self.attention_type_of_layer = model_config_file['attention_type_of_layer']
        self.attention_dropout = model_config_file['attention_dropout']
        self.ffn_dropout = model_config_file['ffn_dropout']
        self.ffn_dim = model_config_file['ffn_dim']
        self.d_out = model_config_file['d_out']
        self.activation = model_config_file['activation']
        self.weight_decay = model_config_file['weight_decay']
        self.momentum = model_config_file['momentum']
        self.lrf = model_config_file['lrf']

        self.ret_all_dataset = model_config_file['read_all_dataset']
        self.table_input = model_config_file['table_train']
        self.dim_k = model_config_file['dim_k']


def load_model_config(config_path):
    if not os.path.exists(config_path):
        AssertionError(f'Filepath {config_path} is not exist')

    full_path_config = os.path.join(os.getcwd(), config_path)
    # path_config = os.path.abspath(dataset_config)

    with open(full_path_config, 'r') as stream:
        model_config = yaml.safe_load(stream)

    return Model_Config(model_config)


# def get_Transformer_Layer(idx, hidden_dim_size, head_num, att_dropout_prob, ffn_dropout_prob, ffn_dim,
#                           activation_fun='relu', type_of_layer='Pre-LN', device='cuda:0', dtype=torch.float64):
#     if type_of_layer == 'Pre-LN':
#         layer_ = nn.ModuleDict({
#             f'norm_layer_{idx}_0': nn.LayerNorm(hidden_dim_size, device=device, dtype=dtype),
#             f'attention_layer_{idx}': nn.MultiheadAttention(hidden_dim_size,
#                                                             head_num,
#                                                             dropout=ffn_dropout_prob,
#                                                             device=device,
#                                                             dtype=dtype),
#             f'Dropout_{idx}_0': nn.Dropout(ffn_dropout_prob),
#             f'norm_layer_{idx}_1': nn.LayerNorm(hidden_dim_size, device=device, dtype=dtype),
#             # Gate
#             f'Gate_linear_{idx}': nn.Linear(hidden_dim_size, 1, device=device, dtype=dtype, bias=False),
#             f'Gate_act_{idx}': nn.Sigmoid(),
#             # FeedForward
#             f'linear_layer_{idx}_0': nn.Linear(hidden_dim_size, ffn_dim, device=device, dtype=dtype),
#             f'linear_layer_{idx}_1': nn.Linear(ffn_dim, hidden_dim_size, device=device, dtype=dtype),
#             f'Dropout_{idx}_1': nn.Dropout(ffn_dropout_prob),

#         })
#         return layer_
#     elif type_of_layer == 'Post-LN':
#         layer_ = nn.ModuleDict({
#             f'attention_layer_{idx}': nn.MultiheadAttention(hidden_dim_size,
#                                                             head_num,
#                                                             dropout=att_dropout_prob,
#                                                             device=device,
#                                                             dtype=dtype),
#             f'Dropout_{idx}_0': nn.Dropout(ffn_dropout_prob),
#             f'norm_layer_{idx}_0': nn.LayerNorm(hidden_dim_size, device=device, dtype=dtype),
#             # Gate
#             f'Gate_linear_{idx}': nn.Linear(hidden_dim_size, 1, device=device, dtype=dtype, bias=False),
#             f'Gate_act_{idx}': nn.Sigmoid(),
#             # FeedForward
#             f'linear_layer_{idx}_0': nn.Linear(hidden_dim_size, ffn_dim, device=device, dtype=dtype),
#             f'linear_layer_{idx}_1': nn.Linear(ffn_dim, hidden_dim_size, device=device, dtype=dtype),
#             f'Dropout_{idx}_1': nn.Dropout(ffn_dropout_prob),
#             f'norm_layer_{idx}_1': nn.LayerNorm(hidden_dim_size, device=device, dtype=dtype),
#         })
#         return layer_

def get_Transformer_Layer(idx, hidden_dim_size, head_num, att_dropout_prob, ffn_dropout_prob, ffn_dim,
                          activation_fun='relu', type_of_layer='Pre-LN', device='cuda:0', dtype=torch.float64):
    if type_of_layer == 'Pre-LN':
        layer_ = nn.ModuleDict({
            f'norm_layer_{idx}_0': nn.LayerNorm(hidden_dim_size, device=device, dtype=dtype),
            f'attention_layer_{idx}': nn.MultiheadAttention(hidden_dim_size,
                                                            head_num,
                                                            dropout=ffn_dropout_prob,
                                                            device=device,
                                                            dtype=dtype),
            f'Dropout_{idx}_0': nn.Dropout(ffn_dropout_prob),
            f'norm_layer_{idx}_1': nn.LayerNorm(hidden_dim_size, device=device, dtype=dtype),
            # FeedForward
            f'linear_layer_{idx}_0': nn.Linear(hidden_dim_size, ffn_dim, device=device, dtype=dtype),
            f'linear_layer_{idx}_1': nn.Linear(ffn_dim, hidden_dim_size, device=device, dtype=dtype),
            f'Dropout_{idx}_1': nn.Dropout(ffn_dropout_prob),
        })
        return layer_
    elif type_of_layer == 'Post-LN':
        layer_ = nn.ModuleDict({
            f'attention_layer_{idx}': nn.MultiheadAttention(hidden_dim_size,
                                                            head_num,
                                                            dropout=att_dropout_prob,
                                                            device=device,
                                                            dtype=dtype),
            f'Dropout_{idx}_0': nn.Dropout(ffn_dropout_prob),
            f'norm_layer_{idx}_0': nn.LayerNorm(hidden_dim_size, device=device, dtype=dtype),
            # FeedForward
            f'linear_layer_{idx}_0': nn.Linear(hidden_dim_size, ffn_dim, device=device, dtype=dtype),
            f'linear_layer_{idx}_1': nn.Linear(ffn_dim, hidden_dim_size, device=device, dtype=dtype),
            f'Dropout_{idx}_1': nn.Dropout(ffn_dropout_prob),
            f'norm_layer_{idx}_1': nn.LayerNorm(hidden_dim_size, device=device, dtype=dtype),
        })
        return layer_


def get_optimiser(optimiser_name, parameters, lr, weight_decay, momentum, beta=0.5):
    if optimiser_name.lower() == 'adam':
        return torch.optim.Adam(params=parameters, lr=lr, weight_decay=weight_decay)
    elif optimiser_name.lower() == 'sgd':
        return torch.optim.SGD(params=parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        AssertionError('Wrong optimiser name avail optimisers : (Adam, SGD)')


def get_activation_function(act_):
    if act_.lower() == 'relu':
        return F.relu
    elif act_.lower() == 'gelu':
        return F.gelu
    elif act_.lower() == 'selu':
        return F.selu_
    elif act_.lower() == 'leakyrelu':
        return F.leaky_relu
    elif act_.lower() == 'sigmoid':
        return F.sigmoid
    elif act_.lower() == 'tanh':
        return F.tanh


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def get_loss_function(device, loss_function_name='mse', num_classes=3, embedding_size=124, batch_size=64):
    if loss_function_name.lower() == 'mse':
        return torch.nn.MSELoss()
    elif loss_function_name.lower() == 'loss_vae':
        return Loss_VAE()
    else:
        AssertionError(f'Wrong loss function {loss_function_name}')

