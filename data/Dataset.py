import glob
import os.path
import pandas as pd
import numpy as np
import yaml
from torch.utils import data
from data.data_utils import transformation_binary
from sklearn.model_selection import train_test_split


class DataBuilder(data.Dataset):
    def __init__(self, filepath, num_con_columns, num_int_columns, num_bin_columns, categorical_columns, col_idx,
                 labels=[], transformation_bin=True):
        super().__init__()
        if type(filepath) is list and type(col_idx) is list:
            df = pd.DataFrame()
            for file, idx in zip(filepath, col_idx):
                dataset_tuple = pd.read_csv(file).iloc[idx]
                df = pd.concat([dataset_tuple, df])
            self.dataset_tuples = df
        else:
            self.dataset_tuples = pd.read_csv(filepath).iloc[col_idx]
        self.num_continuous_col = num_con_columns
        self.num_int_columns = num_int_columns
        self.num_bin_columns = num_bin_columns
        self.categorical_columns = categorical_columns
        self.labels_class_columns = labels
        self.transformation_bin = True

    def __getitem__(self, idx):
        row_tuple = self.dataset_tuples.iloc[[idx]]
        continuous_tuples = row_tuple[self.num_continuous_col].values.flatten().tolist()
        num_int_tuples = row_tuple[self.num_int_columns].values.flatten().tolist()
        bin_tuples = row_tuple[self.num_bin_columns].values.flatten().tolist()
        cat_tuples = row_tuple[self.categorical_columns].values.flatten().tolist()
        if self.transformation_bin:
            bin_tuples = transformation_binary(tuples=bin_tuples)
        continuous_tuples_np = np.asarray(continuous_tuples, dtype=np.float64)
        continuous_tuples_np = np.nan_to_num(continuous_tuples_np)

        num_int_tuples_np = np.asarray(num_int_tuples, dtype=np.float64)
        num_int_tuples_np = np.nan_to_num(num_int_tuples_np)

        bin_tuples_np = np.asarray(bin_tuples, dtype=np.float64)
        bin_tuples_np = np.nan_to_num(bin_tuples_np)

        if self.labels_class_columns is []:
            return continuous_tuples_np, num_int_tuples_np, bin_tuples_np, cat_tuples
        else:
            if len(self.labels_class_columns) > 1:
                class_tuple = row_tuple[self.labels_class_columns].values.flatten().tolist()
            else:
                class_tuple = row_tuple[self.labels_class_columns].values.flatten().tolist()
            class_tuple_np = np.asarray(class_tuple, dtype=np.float64)
            return continuous_tuples_np, num_int_tuples_np, bin_tuples_np, cat_tuples, class_tuple_np

    def __len__(self):
        return self.dataset_tuples.shape[0]


class Dataset:

    def __init__(self, data_conf, return_labels=True, split=True, train=0.75, test=0.15, val=0.1):
        self.datasets_dict = data_conf
        self.d_cat = 0
        self.d_num_con = 0
        self.d_num_int = 0
        self.d_bin = 0
        self.categorical_columns = []
        self.find_d_input()
        # self.add_categorical_columns()
        self.split = split
        if split:
            self.set_train_test_idx(test=test,
                                    train=train,
                                    val=val)
        for dataset_name, dataset in self.datasets_dict.items():
            self.set_paths(dataset_name=dataset_name, filepath=dataset['path'])
    # def add_categorical_columns(self):

    def find_d_input(self):

        for dataset_name, dataset in self.datasets_dict.items():
            # print(dataset)
            self.d_cat = np.maximum(self.d_cat, len(dataset['text']))
            self.d_bin = np.maximum(self.d_bin, len(dataset['binary']))
            self.d_num_con = np.maximum(self.d_num_int, len(dataset['numerical_continues']))
            self.d_num_con = np.maximum(self.d_num_int, len(dataset['numerical_integer']))

    def add_dataset(self, dataset_name, cat_columns, num_con_columns, num_int_columns, bin_columns):
        dataset_ = {"numerical_continues": num_con_columns,
                    "numerical_integer": num_int_columns,
                    "binary": bin_columns,
                    "text": cat_columns}
        self.datasets_dict[dataset_name] = dataset_

    def set_paths(self, filepath, dataset_name):
        if os.path.isfile(filepath):
            return
        else:
            files_ = os.listdir(filepath)
            paths_ = []
            for file in files_:
                filepath_ = os.path.join(filepath, file)
                paths_.append(filepath_)
            self.datasets_dict[dataset_name]['path'] = paths_

    def set_train_test_idx(self, train, test, val):
        for name, dataset in self.datasets_dict.items():
            dataset_path = dataset['path']
            if os.path.isfile(dataset_path):
                with open(dataset_path) as fp:
                    count = 0
                    for _ in fp:
                        count += 1
                dataset_idx = np.arange(start=0, stop=count - 1, step=1, dtype=np.int_)
                train_idx, test_idx = train_test_split(dataset_idx, train_size=train, test_size=(test + val))
                if val > 0:
                    test_idx, val_idx = train_test_split(test_idx, test_size=val)
                    dataset['val_idx'] = val_idx
                dataset['train_idx'] = train_idx
                dataset['test_idx'] = test_idx

                self.datasets_dict[name] = dataset
            else:
                train_idx_lst = []
                test_idx_lst = []
                val_idx_lst = []
                # Is a directory

                files_ = os.listdir(dataset_path)

                for file in files_:
                    filepath = os.path.join(dataset_path, file)
                    with open(filepath) as fp:
                        count = 0
                        for _ in fp:
                            count += 1
                    dataset_idx = np.arange(start=0, stop=count - 1, step=1, dtype=np.int_)
                    train_idx, test_idx = train_test_split(dataset_idx, train_size=train, test_size=(test + val))
                    if val > 0:
                        test_idx, val_idx = train_test_split(test_idx, test_size=val)
                        val_idx_lst.append(val_idx)

                    test_idx_lst.append(test_idx)
                    train_idx_lst.append(train_idx)
                dataset['train_idx'] = train_idx_lst
                dataset['test_idx'] = test_idx_lst
                if val > 0:
                    dataset['val_idx'] = val_idx_lst

    def get_dataset_class_labels(self, dataset_name, class_column):
        dataset_labels = pd.read_csv(self.datasets_dict[dataset_name]['path'])[class_column]
        dataset_labels = list(set(dataset_labels.values.flatten().tolist()))
        return dataset_labels

    def get_Dataset_dataloader(self, dataset_name):
        dataset_path = self.datasets_dict[dataset_name]['path']
        num_con_columns = self.datasets_dict[dataset_name]['numerical_continues']
        num_int_columns = self.datasets_dict[dataset_name]['numerical_integer']
        num_binary = self.datasets_dict[dataset_name]['binary']
        categorical_columns = self.datasets_dict[dataset_name]['text']
        class_columns = self.datasets_dict[dataset_name]['class']

        train_idx = self.datasets_dict[dataset_name]['train_idx']
        test_idx = self.datasets_dict[dataset_name]['test_idx']
        val_idx = self.datasets_dict[dataset_name]['val_idx']

        train_data_builder = DataBuilder(filepath=dataset_path,
                                         num_con_columns=num_con_columns,
                                         num_int_columns=num_int_columns,
                                         num_bin_columns=num_binary,
                                         categorical_columns=categorical_columns,
                                         labels=class_columns,
                                         col_idx=train_idx)
        test_data_builder = DataBuilder(filepath=dataset_path,
                                        num_con_columns=num_con_columns,
                                        num_int_columns=num_int_columns,
                                        num_bin_columns=num_binary,
                                        categorical_columns=categorical_columns,
                                        labels=class_columns,
                                        col_idx=val_idx)
        val_data_builder = DataBuilder(filepath=dataset_path,
                                       num_con_columns=num_con_columns,
                                       num_int_columns=num_int_columns,
                                       num_bin_columns=num_binary,
                                       categorical_columns=categorical_columns,
                                       labels=class_columns,
                                       col_idx=test_idx)

        return train_data_builder, test_data_builder, val_data_builder


def load_data(dataset_config, ):
    if not os.path.exists(dataset_config):
        AssertionError(f'Filepath {dataset_config} is not exist')

    full_path_config = os.path.join(os.getcwd(), dataset_config)
    # path_config = os.path.abspath(dataset_config)

    with open(full_path_config, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    return data_loaded
