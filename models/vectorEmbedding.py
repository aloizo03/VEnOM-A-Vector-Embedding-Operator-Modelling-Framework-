import math
import typing as ty
from pathlib import Path

from torch.xpu import device
import torch.nn.functional as F

from models.utils import get_Transformer_Layer, get_activation_function
import torch
import torch.nn as nn
import torch.nn.init as nn_init
from torch.autograd import Variable


class Decoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 d_token: int,
                 n_layers: int,
                 head_num: int,
                 attention_dropout: float,
                 ffn_dropout: float,
                 ffn_dim: int,
                 activation_fun,
                 d_out: int,
                 device, ):
        super().__init__()
        self.device = device
        self.input_layer_dec = nn.Linear(input_dim, d_token, device=self.device, bias=True, dtype=torch.float64)

        self.Transformer_layers = Embedding_Transformer_Layers(num_layer=n_layers,
                                                               hidden_dim=d_token,
                                                               head_num=head_num,
                                                               ffn_dim=ffn_dim,
                                                               att_dropout_prob=attention_dropout,
                                                               ffn_dropout_prob=ffn_dropout,
                                                               type_of_layer='Pre-LN',
                                                               device=self.device)
        self.out_layer_dec = nn.Linear(d_token, d_out, device=self.device, bias=True, dtype=torch.float64)

        self.activation_fun = activation_fun

    def forward(self, z):
        z_out = self.input_layer_dec(z)
        z_out = self.activation_fun(z_out)
        z_transform = self.Transformer_layers(z_out)
        
        decoded = self.out_layer_dec(self.activation_fun(z_transform))

        return decoded


class Embedding_Transformer_Layers(nn.Module):
    def __init__(self, num_layer, hidden_dim, head_num, ffn_dim=128,
                 att_dropout_prob=0.1, ffn_dropout_prob=0.1, activation='relu',
                 type_of_layer='Pre-LN', device='cuda'):
        super().__init__()
        if num_layer < 1:
            AssertionError('The number of layers for the transformer cannot be zero and bellow!!')
        self.transformer_layers = nn.ModuleList([])
        self.type_of_layer = type_of_layer
        self.activation_function = get_activation_function(act_=activation)
        for idx in range(num_layer):
            layer_ = get_Transformer_Layer(idx=idx,
                                           hidden_dim_size=hidden_dim,
                                           head_num=head_num,
                                           ffn_dim=ffn_dim,
                                           att_dropout_prob=att_dropout_prob,
                                           ffn_dropout_prob=ffn_dropout_prob,
                                           type_of_layer=type_of_layer,
                                           device=device)

            self.transformer_layers.append(layer_)

    def forward(self, x_embd, x_embd_mask=None):
        encoded_embd = x_embd
        for idx, layer_ in enumerate(self.transformer_layers):
            if self.type_of_layer == 'Pre-LN':
                # Execute Pre layer Norm
                x_norm = layer_[f'norm_layer_{idx}_0'](encoded_embd)
                x_attn, x_attn_mask = layer_[f'attention_layer_{idx}'](x_norm, x_norm, x_norm, attn_mask=x_embd_mask)
                x_attn = layer_[f'Dropout_{idx}_0'](x_attn)

                # Fully Forward
                x_attn_norm = layer_[f'norm_layer_{idx}_1'](x_attn)
                x_ffn_0 = layer_[f'linear_layer_{idx}_0'](x_attn_norm)
                x_ffn_1 = layer_[f'linear_layer_{idx}_1'](x_ffn_0)
                x_ffn_1_activated = self.activation_function(x_ffn_1)
                x_ffn_1 = layer_[f'Dropout_{idx}_1'](x_ffn_1_activated)
                encoded_embd = x_ffn_1
            else:
                # Execute Post layer Norm
                x_attn, x_attn_mask = layer_[f'attention_layer_{idx}'](encoded_embd, encoded_embd, encoded_embd,
                                                                       attn_mask=x_embd_mask)
                x_attn = layer_[f'Dropout_{idx}_0'](x_attn)
                x_norm_attn = layer_[f'norm_layer_{idx}_0'](x_attn)

                # Fully Forward
                x_ffn_0 = layer_[f'linear_layer_{idx}_0'](x_norm_attn)
                x_ffn_1 = layer_[f'linear_layer_{idx}_1'](x_ffn_0)
                x_ffn_1_activated = self.activation_function(x_ffn_1)
                x_ffn_1 = layer_[f'Dropout_{idx}_1'](x_ffn_1_activated)
                x_attn_norm = layer_[f'norm_layer_{idx}_1'](x_ffn_1)
                encoded_embd = x_attn_norm

        return encoded_embd

# Will be used from
class Numerical_Embedding(nn.Module):
    def __init__(self, dim, dim_numerical, d_tokens, activation, bias=True, device='cuda'):
        super().__init__()
        dim_input = dim_numerical

        # Embeddings
        self.weights = nn.Parameter(torch.randn(dim_input, dim, device=device))
        print(self.weights.shape)
        self.bias = nn.Parameter(torch.randn(dim_input, dim, device=device)) if bias else None

        self.activation = activation
        self.linear_layer = nn.Linear(dim, d_tokens, device=device, bias=True, dtype=torch.float64)

        nn_init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x):
        x_num = torch.cat([torch.ones(x.shape, device=x.device)]
                          + ([] if x is None else [x]), dim=1)

        x_out = self.weights[:x_num.shape[1]] * x_num[:, :, None]
        if self.bias is not None:
            bias = torch.cat(
                [torch.zeros(1, self.bias.shape[1], device=x.device),
                 self.bias]
            )
            x_out = x_out + bias[:x_out.shape[1]]

        # Check with that in KLD
        # x_out = self.linear_layer(self.activation(x_out))
        return x_out


class Num2Vec(nn.Module):
    def __init__(self,
                 d_numerical,  # float values
                 # transformer
                 n_layers: int,
                 d_token: int,
                 head_num: int,
                 attention_dropout: float,
                 ffn_dropout: float,
                 ffn_dim: int,
                 activation: str,
                 #
                 d_out: int,
                 train: bool,
                 input_table=True,
                 d_k=100,
                 ret_all_dataset=True,
                 distribution='normal'):

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.distribution = distribution

        self.dim_numerical = d_numerical
        self.d_out = d_out
        self.ret_all_dataset = ret_all_dataset

        self.d_tokens = d_token

        self.input_table = input_table
        if input_table:
            # Can be whole dataset or parts from a dataset
            self.dim_k = d_k
            self.dim_numerical = d_numerical * d_k
        self.activation_fun = get_activation_function(act_=activation)

        self.numerical_embedding = Numerical_Embedding(dim=d_token,
                                                       dim_numerical=self.dim_numerical,
                                                       d_tokens=self.d_tokens,
                                                       bias=True,
                                                       device=self.device,
                                                       activation=self.activation_fun)

        self.Transformer_layers = Embedding_Transformer_Layers(num_layer=n_layers,
                                                               hidden_dim=d_token,
                                                               head_num=head_num,
                                                               ffn_dim=ffn_dim,
                                                               att_dropout_prob=attention_dropout,
                                                               ffn_dropout_prob=ffn_dropout,
                                                               type_of_layer='Pre-LN',
                                                               device=self.device)

        # Vector μ and σ
        self.out_layer_m = nn.Linear(d_token, d_out, device=self.device, bias=True, dtype=torch.float64)
        self.out_layer_s = nn.Linear(d_token, d_out, device=self.device, bias=True, dtype=torch.float64)

        # Create decoder
        self.train_ = train
        self.decoder = Decoder(input_dim=d_out,
                               d_token=d_token,
                               ffn_dim=ffn_dim,
                               head_num=head_num,
                               ffn_dropout=ffn_dropout,
                               n_layers=n_layers,
                               d_out=self.dim_numerical,
                               device=self.device,
                               activation_fun=self.activation_fun,
                               attention_dropout=attention_dropout)

    def re_parameterize(self, mu, logvar):
        if self.train_:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        # x_flatten = torch.flatten(x, start_dim=1)

        num_embeddings = self.numerical_embedding(x)
        num_embeddings = self.activation_fun(num_embeddings)
        
        transformed_embeddings = self.Transformer_layers(num_embeddings)
        activated_transformed_embeddings = self.activation_fun(transformed_embeddings)

        out_vector_m = self.out_layer_m(activated_transformed_embeddings[:, 0, :])
        out_vector_s = F.sigmoid(self.out_layer_s(activated_transformed_embeddings[:, 0, :]))

        z = self.re_parameterize(out_vector_m, out_vector_s)

        if self.train_:
            decoded = self.decoder(z)
            # TODO: Add to return the z so add to the loss function 
            return decoded, out_vector_m, out_vector_s

        return None, out_vector_m, out_vector_s
