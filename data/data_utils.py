import numpy as np
import pandas as pd
import torch


def collator_fn(data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_cat_values = []
    col_float, col_int, col_bin, col_cat, col_labels = data[0]
    out_num_float_values = torch.empty(size=(len(data), col_float.shape[0]), dtype=torch.float64, device=device)
    out_num_int_values = torch.empty(size=(len(data), col_int.shape[0]), dtype=torch.int, device=device)
    out_num_bin_values = torch.empty(size=(len(data), col_bin.shape[0]), dtype=torch.float64, device=device)
    out_labels = torch.empty(size=(len(data), len(col_labels)), dtype=torch.float64, device=device)

    for idx, batch_ in enumerate(data):
        continuous_tuples_np, num_int_tuples_np, bin_tuples_np, cat_tuples, class_tuple = batch_
        out_cat_values.append(cat_tuples)
        out_num_float_values[idx] = torch.tensor(continuous_tuples_np, dtype=torch.float64, device=device)
        out_num_int_values[idx] = torch.tensor(num_int_tuples_np, dtype=torch.float64, device=device)
        out_num_bin_values[idx] = torch.tensor(bin_tuples_np, dtype=torch.float64, device=device)
        out_labels[idx] = torch.tensor(class_tuple, dtype=torch.float64, device=device)
    return out_num_float_values, out_num_int_values, out_num_bin_values, out_cat_values, out_labels


def transformation_binary(tuples):
    print(tuples)
    tuples_set = set(tuples.tolist())
    print(tuples_set)
    # transform_tuples = [int(x) for x in tuples]
    return transform_tuples
