import glob
import os.path
import pandas as pd
import numpy as np
import yaml
from torch.utils import data
from data.data_utils import transformation_binary
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


class DataBuilder(data.Dataset):
    def __init__(self, filepath, columns, col_idx,
                 labels=[], norm=True, augmentation=True, ret_labels=False):
        super().__init__()
        # Chnage it to take per file and when finish to open a new file
        if type(filepath) is list and type(col_idx) is list:
            df = pd.DataFrame()
            for file, idx in zip(filepath, col_idx):
                dataset_tuple = pd.read_csv(file).iloc[idx]
                df = pd.concat([df, dataset_tuple])
            self.dataset_tuples = df
        else:
            self.dataset_tuples = pd.read_csv(filepath).iloc[col_idx]
        self.columns = columns
        self.labels_class_columns = labels
        self.transformation_bin = False
        self.normalise_data = norm
        self.augmentation = augmentation
        self.ret_labels = ret_labels

    def __getitem__(self, idx):
        row_tuple = self.dataset_tuples.iloc[[idx]]
        tuples = row_tuple[self.columns].values.flatten().tolist()

        tuples_np = np.asarray(tuples, dtype=np.float64)
        tuples_np = np.nan_to_num(tuples_np)

        if self.normalise_data:
            tuples_np = preprocessing.normalize([tuples_np])[0]

        if self.ret_labels:
            class_tuple = row_tuple[self.labels_class_columns].values.flatten().tolist()
            class_tuple_np = np.asarray(class_tuple, dtype=np.float64)
            return tuples_np, class_tuple_np
        else:
            return tuples_np

    def __len__(self):
        return self.dataset_tuples.shape[0]


class Dataset:

    def __init__(self, data_conf, return_labels=True, split=True,
                 train=0.75, test=0.15, val=0.1, norm=False, augment=True):
        self.datasets_dict = data_conf
        self.categorical_columns = []
        self.normalise_data = norm
        self.augmentation = augment

        # self.add_categorical_columns()
        self.split = split
        if split:
            self.set_train_test_idx(test=test,
                                    train=train,
                                    val=val)
        for dataset_name, dataset in self.datasets_dict.items():
            self.set_paths(dataset_name=dataset_name, filepath=dataset['path'])

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
        tuples_columns = self.datasets_dict[dataset_name]['columns']
        class_columns = self.datasets_dict[dataset_name]['class']

        train_idx = self.datasets_dict[dataset_name]['train_idx']
        test_idx = self.datasets_dict[dataset_name]['test_idx']
        val_idx = self.datasets_dict[dataset_name]['val_idx']

        train_data_builder = DataBuilder(filepath=dataset_path,
                                         columns=tuples_columns,
                                         labels=class_columns,
                                         col_idx=train_idx,
                                         norm=self.normalise_data)
        test_data_builder = DataBuilder(filepath=dataset_path,
                                        columns=tuples_columns,
                                        labels=class_columns,
                                        col_idx=val_idx,
                                        norm=self.normalise_data)
        val_data_builder = DataBuilder(filepath=dataset_path,
                                       columns=tuples_columns,
                                       labels=class_columns,
                                       col_idx=test_idx,
                                       norm=self.normalise_data)

        return train_data_builder, test_data_builder, val_data_builder


def load_data(dataset_config, ):
    if not os.path.exists(dataset_config):
        AssertionError(f'Filepath {dataset_config} is not exist')

    full_path_config = os.path.join(os.getcwd(), dataset_config)
    # path_config = os.path.abspath(dataset_config)

    with open(full_path_config, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    return data_loaded
