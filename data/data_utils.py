import numpy as np
import pandas as pd
import torch


def padding(data, dim_, padding_type='constant'):
    return np.pad(data, dim_, padding_type)


def split_sub_tables(df, chunk_size):
    df_chunks = []
    for i in range(0, df.shape[0], chunk_size):
        chunk_df = df.iloc[i:i+chunk_size, :]
        df_chunks.append(chunk_df)

    return df_chunks


def collator_fn(data, data_cat):
    # TODO: Make to stack data with different dimensional inputs
    max_dim_rows = np.max([d.shape[0] for d in data])
    max_dim_col = np.max([d.shape[1] for d in data])
    reshape_data = []
    for d in data:
        zero_pad = np.zeros(shape=(max_dim_rows, max_dim_col), dtype=np.float64)
        zero_pad[:d.shape[0], :d.shape[1]] = d
        zero_pad = zero_pad.flatten()

        tensor_ = torch.tensor(zero_pad, dtype=torch.float64)
        reshape_data.append(tensor_)

    stack_data = torch.stack(reshape_data, dim=0)
    df = data_cat[:].values
    
    return stack_data, df


def transformation_binary(tuples):
    tuples_set = set(tuples.tolist())
    transform_tuples = [int(x) for x in tuples]
    return transform_tuples
