import math
import typing as ty
from pathlib import Path
from models.utils import get_Transformer_Layer, get_activation_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch.distributions.uniform import Uniform
from torch import Tensor
from transformers import BertTokenizer, BertTokenizerFast
import os


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()


# class TrasnformerLayer(nn.Module):
#
#     def __init__(self):
#         super().__init__()


class Embedding_Transformer_Layers(nn.Module):
    def __init__(self, num_layer, hidden_dim, head_num, ffn_dim=128,
                 att_dropout_prob=0.1, ffn_dropout_prob=0.1, activation='relu',
                 type_of_layer='Pre-LN', device='cuda:0'):
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


class Embedding_Categorical_Feature_Embedding(nn.Module):
    def __init__(self, num_of_tokens, hidden_dim_size):
        super().__init__()
        self.device = self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self):
        pass


# Will be used from
class Embedding_Numerical_Feature_Embedding(nn.Module):
    def __init__(self, token_dim, hidden_dim_size, num_of_layers=1, head_num=1, dropout_prob=0.1, init_weights=False,
                 norm=False, bias=False, dtype=torch.float64, activation='relu'):
        super().__init__()
        self.device = self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_of_layers = num_of_layers
        self.num_embeddings = nn.Linear(token_dim, hidden_dim_size, bias=bias, device=self.device, dtype=dtype)

        self.attention_layer_embedding_Transform = get_Transformer_Layer(hidden_dim_size=hidden_dim_size,
                                                                         head_num=head_num,
                                                                         device=self.device,
                                                                         att_dropout_prob=dropout_prob,
                                                                         ffn_dropout_prob=dropout_prob,
                                                                         ffn_dim=hidden_dim_size * 2,
                                                                         idx=0)
        self.type_of_layer = 'Post-LN'
        self.activation_function = get_activation_function(act_=activation)

    def forward(self, x_num):
        x_embd = self.num_embeddings(x_num)
        if self.type_of_layer == 'Pre-LN':
            # Execute Pre layer Norm
            x_norm = layer_[f'norm_layer_0_0'](encoded_embd)
            x_attn, x_attn_mask = layer_[f'attention_layer_0'](x_norm, x_norm, x_norm, attn_mask=x_embd_mask)
            x_attn = layer_[f'Dropout_0_0'](x_attn)

            # Fully Forward
            x_attn_norm = layer_[f'norm_layer_0_1'](x_attn)
            x_ffn_0 = layer_[f'linear_layer_0_0'](x_attn_norm)
            x_ffn_1 = layer_[f'linear_layer_0_1'](x_ffn_0)
            x_ffn_1_activated = self.activation_function(x_ffn_1)
            x_ffn_1 = layer_[f'Dropout_0_1'](x_ffn_1_activated)
            encoded_embd = x_ffn_1
        else:
            # Execute Post layer Norm
            x_attn, x_attn_mask = self.attention_layer_embedding_Transform[f'attention_layer_0'](x_embd, x_embd, x_embd)
            x_attn = self.attention_layer_embedding_Transform[f'Dropout_0_0'](x_attn)
            x_norm_attn = self.attention_layer_embedding_Transform[f'norm_layer_0_0'](x_attn)

            # Fully Forward
            x_ffn_0 = self.attention_layer_embedding_Transform[f'linear_layer_0_0'](x_norm_attn)
            x_ffn_1 = self.attention_layer_embedding_Transform[f'linear_layer_0_1'](x_ffn_0)
            x_ffn_1_activated = self.activation_function(x_ffn_1)
            x_ffn_1 = self.attention_layer_embedding_Transform[f'Dropout_0_1'](x_ffn_1_activated)
            x_attn_norm = self.attention_layer_embedding_Transform[f'norm_layer_0_1'](x_ffn_1)
            encoded_embd = x_attn_norm

        return x_embd + encoded_embd


class Feature_Embedding(nn.Module):

    def __init__(self, token_dim, hidden_dim, dropout_prob=0.1, num_of_layers=1, device='cuda:0', Linear_layer=False):
        super().__init__()
        self.device = device
        self.embedding_categorical = None
        self.embedding_num_continuous = Embedding_Numerical_Feature_Embedding(token_dim=token_dim,
                                                                              hidden_dim_size=hidden_dim,
                                                                              dropout_prob=dropout_prob,
                                                                              num_of_layers=num_of_layers)
        self.embedding_num_int = Embedding_Numerical_Feature_Embedding(token_dim=token_dim,
                                                                       hidden_dim_size=hidden_dim,
                                                                       dropout_prob=dropout_prob,
                                                                       num_of_layers=num_of_layers)

        self.embedding_bin = Embedding_Numerical_Feature_Embedding(token_dim=token_dim,
                                                                   hidden_dim_size=hidden_dim,
                                                                   dropout_prob=dropout_prob,
                                                                   num_of_layers=num_of_layers)

        self.align_layer = nn.Linear(hidden_dim, hidden_dim, bias=False, device=self.device, dtype=torch.float64)

        # if linear_layer:
        #     self.linear_layer_embedding =

    def forward(self, tokenizer_dict):
        num_integer_embeddings = None
        num_continuous_embeddings = None
        num_binary_embeddings = None

        embedding_feature_list = []
        if len(tokenizer_dict['X_Integer']) > 0:
            num_integer_embeddings = self.embedding_num_int(tokenizer_dict['X_Integer'])
            num_integer_embeddings = self.align_layer(num_integer_embeddings)
            embedding_feature_list.append(num_integer_embeddings)

        if len(tokenizer_dict['X_Continues']) > 0:
            num_continuous_embeddings = self.embedding_num_continuous(tokenizer_dict['X_Continues'])
            num_continuous_embeddings = self.align_layer(num_continuous_embeddings)
            embedding_feature_list.append(num_continuous_embeddings)

        if len(tokenizer_dict['X_categorical']) > 0:
            pass

        if len(tokenizer_dict['X_Binary']) > 0:
            num_binary_embeddings = self.embedding_bin(tokenizer_dict['X_Binary'])
            num_binary_embeddings = self.align_layer(num_binary_embeddings)
            embedding_feature_list.append(num_binary_embeddings)

        # Concatenation of all the embedding features
        concat_embedding_features = torch.cat(embedding_feature_list, dim=1)

        return concat_embedding_features


class feature_Extractor(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(self,
                 d_numerical_continues,  # Float values
                 d_numerical_integer,
                 d_binary,
                 categories,
                 d_token,
                 bias=False,
                 use_bert=False):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_bert = use_bert
        self.bias = bias
        # TODO: Fix pretrained bert to work only for categorical data features
        if use_bert:
            if os.path.exists('./pretrained/tokenizer'):
                self.tokenizer = BertTokenizerFast.from_pretrained('./pretrained/tokenizer')
            else:
                self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
                self.tokenizer.save_pretrained('./pretrained/tokenizer')
        else:
            if categories is None:
                d_bias = d_numerical_integer + d_numerical_continues
                self.category_offsets = None
                self.category_embeddings = None
                self.categoryMHA = None
            else:
                d_bias = d_numerical_integer + d_numerical_integer + len(categories)
                categories_offset = torch.tensor([0] + categories[:-1]).cumsum(0)
                # self.register_buffer('category_offsets', categories_offset)
                self.category_embeddings = nn.Embedding(sum(categories), d_token)
                nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
                # print(f'{self.category_embeddings.weight.shape}')

            # TODO: Add dictionary parameters
            self.dict_params_tokenizer = nn.ParameterDict({
                'embedding_int_w': nn.Parameter(Tensor(d_numerical_integer + 1, d_token).to(self.device)),
                'embedding_int_bias': nn.Parameter(Tensor(d_bias, d_token, device=self.device)) if bias else None,
                'embedding_numerical_continues_w': nn.Parameter(
                    Tensor(d_numerical_continues + 1, d_token).to(self.device)),
                'embedding_numerical_continues_bias': nn.Parameter(
                    Tensor(d_bias, d_token, device=self.device)) if bias else None,
                'embedding_binary_w': nn.Parameter(Tensor(d_binary + 1, d_token).to(self.device)),
                'embedding_binary_bias': nn.Parameter(Tensor(d_bias, d_token)) if bias else None
            })
            # self.embedding_int_w = nn.Parameter(Tensor(d_numerical_integer + 1, d_token).to(self.device))
            # self.embedding_int_bias = nn.Parameter(Tensor(d_bias, d_token, device=self.device)) if bias else None
            nn_init.kaiming_uniform_(self.dict_params_tokenizer['embedding_int_w'], a=math.sqrt(5))
            #
            # self.embedding_numerical_continues_w = nn.Parameter(
            #     Tensor(d_numerical_continues + 1, d_token).to(self.device))
            # self.embedding_numerical_continues_bias = nn.Parameter(
            #     Tensor(d_bias, d_token, device=self.device)) if bias else None
            nn_init.kaiming_uniform_(self.dict_params_tokenizer['embedding_numerical_continues_w'], a=math.sqrt(5))
            #
            # self.embedding_binary_w = nn.Parameter(Tensor(d_binary + 1, d_token).to(self.device))
            # self.embedding_binary_bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
            nn_init.kaiming_uniform_(self.dict_params_tokenizer['embedding_binary_w'], a=math.sqrt(5))

            if self.bias:
                nn_init.kaiming_uniform_(self.dict_params_tokenizer['embedding_int_bias'], a=math.sqrt(5))
                nn_init.kaiming_uniform_(self.dict_params_tokenizer['embedding_numerical_continues_bias'],
                                         a=math.sqrt(5))
                nn_init.kaiming_uniform_(self.dict_params_tokenizer['embedding_binary_bias'], a=math.sqrt(5))

    def n_tokens(self):
        return len(self.weight) + (0 if self.category_offsets is None else len(self.category_offsets))

    def forward(self, x_int, x_cat=None, x_bin=None, x_float=None):
        out_dict = {
            "X_Integer": [],
            "X_Continues": [],
            "X_categorical": [],
            "X_Binary": []
        }

        if self.use_bert:
            if x_int is not None and len(x_int) > 0:
                x_int_tensor = torch.tensor(x_int, dtype=torch.int, device=self.device)
                x_int_out = self.tokenizer.tokenize(x_int_tensor,
                                                    padding=True,
                                                    truncation=True,
                                                    add_special_tokens=False, return_tensors='pt')
                out_dict["X_Integer"] = x_int_out

            if x_float is not None and len(x_float) > 0:
                x_float_tensor = torch.tensor(x_float, dtype=torch.float, device=self.device)
                x_float_out = self.tokenizer.tokenize(x_float_tensor,
                                                      padding=True,
                                                      truncation=True,
                                                      add_special_tokens=False, return_tensors='pt')
                out_dict['X_Continues'] = x_float_out
        else:

            if x_int is not None and x_int.shape[0] > 0:
                if torch.is_tensor(x_int):
                    x_int_tensor = x_int.to(self.device)
                else:
                    x_int_tensor = torch.tensor(x_int, dtype=torch.int, device=self.device)
                x_int_tensor = torch.cat([torch.ones(x_int_tensor.shape, device=self.device)]
                                         + ([] if x_int_tensor is None else [x_int_tensor]), dim=-1)

                x_int_out = self.dict_params_tokenizer['embedding_int_w'][None] * x_int_tensor[:, :, None]
                if self.bias:
                    bias = torch.cat(
                        [torch.zeros(1, self.dict_params_tokenizer['embedding_int_bias'].shape[1], device=self.device),
                         self.dict_params_tokenizer['embedding_int_bias']])
                    x_int_out = x_int_out + bias[None]
                out_dict["X_Integer"] = x_int_out.type(torch.double)

            if x_float is not None and x_float.shape[0] > 0:
                if torch.is_tensor(x_float):
                    x_float_tensor = x_float.to(self.device)
                else:
                    x_float_tensor = torch.tensor(x_float, dtype=torch.float, device=self.device)
                x_float_tensor = torch.cat([torch.ones(x_float_tensor.shape, device=self.device)] + (
                    [] if x_float_tensor is None else [x_float_tensor]), dim=-1)
                x_float_out = self.dict_params_tokenizer['embedding_int_w'][None] * x_float_tensor[:, :, None]

                if self.bias:
                    bias = torch.cat(
                        [torch.zeros(1, self.dict_params_tokenizer['embedding_numerical_continues_bias'].shape[1],
                                     device=self.device),
                         self.dict_params_tokenizer['embedding_numerical_continues_bias']])
                    x_float_out = x_int_out + bias[None]
                out_dict["X_Continues"] = x_float_out.type(torch.double)

            if x_bin is not None and x_bin.shape[0] > 0:
                if torch.is_tensor(x_bin):
                    x_bin_tensor = x_bin.to(self.device)
                else:
                    x_bin_tensor = torch.tensor(x_bin, dtype=torch.int, device=self.device)
                x_bin_tensor = torch.cat([torch.ones(x_bin_tensor.shape, device=self.device)] + (
                    [] if x_bin_tensor is None else [x_bin_tensor]), dim=1)
                x_bin_out = self.dict_params_tokenizer['embedding_int_w'][None] * x_bin_tensor[:, :, None]

                if self.bias:
                    bias = torch.cat([torch.zeros(1, self.dict_params_tokenizer['embedding_binary_bias'].shape[1],
                                                  device=self.device),
                                      self.dict_params_tokenizer['embedding_binary_bias']])
                    x_bin_out = x_bin_out + bias[None]

                out_dict["X_Binary"] = x_bin_out.type(torch.double)

        return out_dict


class VectorEmbedding_Transformer(nn.Module):

    def __init__(self,
                 # tokenizer
                 d_numerical_continues,  # float values
                 d_numerical_integer,  # integer values
                 categories: ty.Optional[ty.List[int]],  # categorical values
                 d_binary,  # binary_values
                 token_bias,
                 use_pretrained_bert,
                 # transformer
                 n_layers: int,
                 d_token: int,
                 head_num: int,
                 attention_dropout: float,
                 attention_hidden_dim: int,
                 ffn_dropout: float,
                 ffn_dim: int,
                 activation: str,
                 #
                 d_out: int,
                 corruption_rate=0.5):
        super().__init__()
        # Feature extractor for feature tokenizer
        # Later work see if we can have BERT head or TaBERT
        self.corruption_rate = corruption_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer_fe = feature_Extractor(d_numerical_continues=d_numerical_continues,
                                              d_numerical_integer=d_numerical_integer,
                                              categories=categories,
                                              d_binary=d_binary,
                                              d_token=d_token,
                                              bias=token_bias,
                                              use_bert=use_pretrained_bert)

        self.feature_processing = Feature_Embedding(token_dim=d_token,
                                                    hidden_dim=attention_hidden_dim,
                                                    dropout_prob=attention_dropout,
                                                    num_of_layers=1,
                                                    device=self.device)

        self.transformer_layers = Embedding_Transformer_Layers(num_layer=n_layers,
                                                               hidden_dim=attention_hidden_dim,
                                                               head_num=head_num,
                                                               ffn_dim=ffn_dim,
                                                               att_dropout_prob=attention_dropout,
                                                               ffn_dropout_prob=ffn_dropout,
                                                               type_of_layer='Pre-LN',
                                                               device=self.device)

        self.out_head_layer = nn.Linear(attention_hidden_dim, d_out, device=self.device, bias=False, dtype=torch.float64)
        self.activation_fun = get_activation_function(act_=activation)

    # def change_out_head_dim(self, out_dim, bias=False):
    #     print(self.out_head_layer.__sizeof__())

    def forward(self, x_int, x_float, x_bin, x_cat):
        # x_int, x_cat = None, x_bin = None, x_float = None
        tokenized = self.tokenizer_fe(x_int, x_cat, x_bin, x_float)

        concat_embedding_features = self.feature_processing(tokenized)
        encoded_embedding = self.transformer_layers(concat_embedding_features)
        out_x_projected = self.out_head_layer(encoded_embedding[:, 1, :])

        return out_x_projected
