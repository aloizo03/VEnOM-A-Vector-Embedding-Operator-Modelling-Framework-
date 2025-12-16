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
        self.text_tokenizer = model_config_file['text_tokenizer']

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

#
# class KAN_Transformer(nn.Module):
#
#     def __init__(self, vocabulary_size, hidden_size, head_num, win_size, d_ff, num_experts, n_experts_per_tokens,
#                  n_blocks,
#                  max_seq_len):
#         super().__init__()
#
#         self.embedding = nn.Embedding(vocabulary_size, hidden_size)
#         head_dim = hidden_size // head_num
#         rot_matrix = get_rotation_matrix(head_dim, max_seq_len, 10000.)
#
#         self.blocks = nn.ModuleList([
#             KAN_Block(hidden_size, head_num, win_size, d_ff, num_experts, n_experts_per_tokens, rot_matrix) for _ in
#             range(n_blocks)
#         ])
#         self.out = nn.Linear(hidden_size, vocabulary_size)
#
#     def forward(self, x):
#         emb_vec = self.embedding(x)
#         print(emb_vec.size())
#         for b in self.blocks:
#             emb_vec = b(emb_vec)
#         return self.out(emb_vec)
#
#
# class MoeKANLayer(nn.Module):
#     def __init__(self, hidden_size, d_ff, num_experts, num_experts_per_token):
#         super().__init__()
#         self.num_experts = num_experts
#         self.num_experts_per_token = num_experts_per_token
#         self.experts = nn.ModuleList([
#             FeedForward(hidden_size, d_ff) for _ in range(num_experts)
#         ])
#         self.gate = nn.Linear(hidden_size, num_experts, bias=False)
#
#     def forward(self, x):
#         gate_logits = self.gate(x)
#         weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_token)
#         weights = F.softmax(weights, dim=-1, dtype=torch.float).to(x.dtype)
#         out = torch.zeros_like(x)
#
#         for i, expert in enumerate(self.experts):
#             batch_idx, token_idx, topk_idx = torch.where(selected_experts == i)
#             weight = weights[batch_idx, token_idx, topk_idx, None]
#             out[batch_idx, token_idx] += weight * expert(x[batch_idx, token_idx])
#         return out
#
#
# class FeedForward(nn.Module):
#     def __init__(self, hidden_size, d_ff):
#         super().__init__()
#         self.w1 = KANLinear(hidden_size, d_ff)
#         self.w2 = KANLinear(hidden_size, d_ff)
#         self.w3 = KANLinear(d_ff, hidden_size)
#
#     def forward(self, x):
#         return self.w3(self.w1(x) * self.w2(x))
#
#
# class KAN_Block(nn.Module):
#
#     def __init__(self, hidden_size, head_nums, win_size, d_ff, num_experts, num_experts_per_token, rot_matrix):
#         super().__init__()
#         self.norm_1 = RMS_Norm(hidden_size)
#         self.norm_2 = RMS_Norm(hidden_size)
#         # TODO: Fix There
#         self.attention = KAN_MultiHeadAttention(hidden_size, head_nums, win_size, rot_matrix)
#         self.moe = MoeKANLayer(hidden_size, d_ff, num_experts, num_experts_per_token)
#
#     def forward(self, x):
#         x1 = self.attention(self.norm_1(x))
#         x = x + x1
#         x2 = self.moe(self.norm_2(x))
#         out = x + x2
#         return out
#
#
# class RMS_Norm(nn.Module):
#
#     def __init__(self, hidden_size, eps=1e-6):
#         super().__init__()
#         self.eps = eps
#         self.hidden_size = hidden_size
#         self.weights = nn.Parameter(torch.ones(hidden_size))
#
#     def _norm(self, x):
#         return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
#
#     def forward(self, x):
#         output = self._norm(x.float()).type_as(x)
#         return output * self.weights
#
#
# class KAN_MultiHeadAttention(nn.Module):
#     def __init__(self, hidden_dim, head_num, win_size, rotation_matrix):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.head_num = head_num
#         self.head_dim = hidden_dim // head_num
#         self.rotation_matrix = rotation_matrix
#
#         self.layer_ = KANLinear(self.hidden_dim, self.hidden_dim * 3)  # Error Here
#
#         self.out_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.position_embeddings = RoPE(rotation_matrix)
#
#     def forward(self, x):
#         batch_size = 1
#         seq_len, hidden_size = x.size()
#
#         kan_qkv = self.layer_(x)  # Error here
#         kan_qkv.reshape(batch_size, seq_len, self.head_num, 3 * self.head_dim)
#         kan_qkv.transpose(1, 2)
#         q, k, v = kan_qkv.chunk(3, dim=-1)
#         q, k = self.position_embeddings(q, k)
#
#         scores = torch.matmul(q, k.transpose(2, 3))
#         scores = scores / (self.head_dim ** 0.5)
#         attention = F.softmax(scores, dim=-1)
#         context = torch.matmul(attention, v)
#         context = context.transpose(1, 2)
#         # context = context.reshape(batch_size, seq_len, hidden_size)
#
#         return self.out_layer(context)
#
#
# class RoPE(nn.Module):
#     def __init__(self, rotation_matrix):
#         super().__init__()
#         self.rotation_matrix = rotation_matrix
#
#     def forward(self, queries, keys):
#         batch_size, head_num, seq_len, head_dim = queries.size()
#
#         queries = queries.reshape(batch_size, head_num, seq_len, head_dim // 2, 2)
#         keys = keys.reshape(batch_size, head_num, seq_len, head_dim // 2, 2)
#
#         queries_complex = torch.view_as_complex(queries)
#         key_complex = torch.view_as_complex(keys)
#
#         rot_matrix = self.rotation_matrix[:seq_len]
#
#         queries_rot = queries_complex * rot_matrix
#         keys_rot = key_complex * rot_matrix
#
#         new_queries = torch.view_as_real(queries_rot)
#         new_keys = torch.view_as_real(keys_rot)
#
#         new_queries = new_queries.reshape(batch_size, head_num, seq_len, head_dim)
#         new_keys = new_keys.reshape(batch_size, head_num, seq_len, head_dim)
#
#         return new_queries, new_keys
#
#
# def get_rotation_matrix(dim, context_size, period):
#     freq = 1.0 / (period ** (torch.arange(0, dim, 2) / dim))
#     token_indexes = torch.arange(context_size)
#     thetas = torch.outer(token_indexes, freq).float()
#     return torch.polar(torch.ones_like(thetas), thetas)
#
#
# def main():
#     model = KAN_Transformer(vocabulary_size=10,
#                             hidden_size=20,
#                             head_num=10,
#                             win_size=4,
#                             d_ff=4,
#                             num_experts=3,
#                             n_experts_per_tokens=10,
#                             n_blocks=3,
#                             max_seq_len=10)
#
#     x = torch.randint(3, 10, (10,))
#     print(x)
#     out = model(x)
#
#
# if __name__ == '__main__':
#     main()
