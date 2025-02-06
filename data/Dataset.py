import glob
import os.path
import pandas as pd
import numpy as np
import yaml
from torch.utils import data
from data.data_utils import transformation_binary, padding, split_sub_tables
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import warnings
from utils_.utils import check_path

warnings.filterwarnings("ignore")


class DataBuilder(data.Dataset):
    def __init__(self, filepath, dir_path, columns, col_idx,
                 labels=[], norm=True, padding_=True, padding_dim=(2, 2),
                 augmentation=True, ret_labels=False,
                 ret_table=True, k_dim=40, ret_all_dataset=True):
        super().__init__()
        # Change it to take per file and when finish to open a new file

        if ret_table and ret_all_dataset:
            self.dataset_tuples = []
        if type(filepath) is list and type(col_idx) is list:
            if ret_all_dataset:
                for file in col_idx:
                    dataset_tuple = pd.read_csv(file)
                    self.dataset_tuples.append(dataset_tuple)
            else:
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
        self.padding = padding_
        self.pad_dim = padding_dim
        self.ret_all_dataset = ret_all_dataset

        self.ret_table = ret_table
        self.k_dim = k_dim

        if ret_table and not ret_all_dataset:
            # chunk_size = np.floor(self.dataset_tuples.shape[0] / self.k_dim)
            self.dataset_tuples = split_sub_tables(df=self.dataset_tuples, chunk_size=self.k_dim)

    def __getitem__(self, idx):
        if self.ret_table:
            row_tuples = self.dataset_tuples[idx]

            if self.ret_labels:
                tuples = row_tuples.values
            else:
                tuples = row_tuples[self.columns[idx]].values
            # if tuples.shape[0] < self.k_dim:
            #     zero_pad = np.zeros(shape=(self.k_dim, tuples.shape[1]))
            #     zero_pad[:tuples.shape[0], :] = tuples
            #     tuples = zero_pad
        else:
            row_tuples = self.dataset_tuples.iloc[[idx]]
            tuples = row_tuples[self.columns].values.flatten().tolist()

        tuples_np = np.asarray(tuples, dtype=np.float64)
        tuples_np = np.nan_to_num(tuples_np)

        if self.normalise_data:
            if self.ret_table:
                if len(tuples_np.shape) < 2:
                    tuples_np = preprocessing.normalize([tuples_np])[0]
                else:
                    tuples_np = preprocessing.normalize(tuples_np)
            else:
                tuples_np = preprocessing.normalize([tuples_np])[0]

        if self.ret_labels:
            if self.ret_all_dataset:
                return tuples_np
            else:
                class_tuple = row_tuples[self.labels_class_columns].values.flatten().tolist()
                class_tuple_np = np.asarray(class_tuple, dtype=np.float64)
                return tuples_np, class_tuple_np
        else:
            return tuples_np

    def __len__(self):
        if self.ret_table:
            return len(self.dataset_tuples)
        else:
            return self.dataset_tuples.shape[0]


class DataBuilder_Pipeline(data.Dataset):

    def __init__(self, datasets_filepaths, norm=True, ret_Table=True, ret_label=False, k_dim=5, ret_all_dataset=True,
                 create_operator=False, ret_class=False, Vectors=None):
        self.datasets_data = []
        self.datasets_filename = datasets_filepaths
        self.norm = norm

        self.ret_label = ret_label
        self.ret_table = ret_Table
        self.ret_class = ret_class

        self.k_dim = k_dim
        self.data_len = 0
        self.ret_all_dataset = ret_all_dataset

        self.create_operator = create_operator
        self.vectors = Vectors

        if type(self.datasets_filename) is list:
            self.idx_to_filename = {idx: filename for idx, filename in enumerate(self.datasets_filename)}

            if self.create_operator:
                df = pd.DataFrame()
                for filename in self.datasets_filename:
                    df = pd.concat([df, pd.read_csv(filename)])
                self.datasets_data = df
                self.data_len = len(self.datasets_data)
            else:
                for filename in self.datasets_filename:
                    data_ = pd.read_csv(filename)
                    if self.ret_all_dataset:
                        self.datasets_data.append(data_)
                    else:
                        data_chunks = split_sub_tables(df=data_, chunk_size=self.k_dim)
                        self.datasets_data.append(data_chunks)
                        self.data_len = self.data_len + len(data_chunks)
                if self.ret_all_dataset:
                    self.data_len = len(self.datasets_data)
        else:
            self.idx_to_filename = {idx: filename for idx, filename in enumerate([self.datasets_filename])}

            if self.create_operator:
                # print(self.datasets_filename)
                self.datasets_data = pd.read_csv(self.datasets_filename)
                self.data_len = self.datasets_data.shape[0]
            else:
                self.datasets_data.append(pd.read_csv(self.datasets_filename))
                if self.ret_table:
                    self.data_len = len(self.datasets_data)
                else:
                    self.data_len = self.datasets_data[0].shape[0]

        self.current_dataset_id = 0
        self.data_idx = 0

    def get_dataset_name(self, idx):
        return self.idx_to_filename[idx]

    def get_all_data_sr(self, query='last'):
        vectors_list = []
        labels_list = []

        if query == 'last':
            if type(self.datasets_data is list):
                for data_, vector_ in zip(self.datasets_data, self.vectors):
                    dataset_data = data_.dropna()
                    class_labels = dataset_data.loc[:, data_.columns.str.startswith("class")]
                    last_class_label = class_labels.tail(1)

                    vectors_list.append(vector_)
                    labels_list.append(last_class_label)

        vec_np = np.asarray(vectors_list)
        label_np = np.asarray(labels_list)

        label_np = label_np.reshape(-1, 1)
        print(label_np.shape)
        return vec_np, label_np

        # dataset_data = data_.dropna()
        # class_labels =  dataset_data.loc[:, data_.columns.str.startswith("class")]
        #             row_tuples =  dataset_data.drop(columns=list(class_labels.columns))

        #             last_row_np = np.asarray(row_tuples.tail(1), dtype=np.float64)
        #             last_class_label = np.asarray(class_labels.tail(1), dtype=np.float64)

        #             row_tuples = row_tuples.drop(row_tuples.index[len(row_tuples)-1])
        #             class_labels = class_labels.drop(class_labels.index[len(class_labels)-1])

        #             class_labels = np.asarray(class_labels, dtype=np.float64)

        #             tuples_np = np.asarray(row_tuples, dtype=np.float64)
        #             tuples_np = np.nan_to_num(tuples_np)

        #             if self.norm:
        #                 tuples_np = preprocessing.normalize(tuples_np)
        #                 last_row_np =  preprocessing.normalize(last_row_np)

        #             ret_array.append(tuples_np)
        #             ret_array_label.append(class_labels)

    def get_all_data_per_dataset(self, query='last', concat=True):
        data_set_lst = []
        labels_lst = []

        last_row_data_lst = []
        last_label_lst = []

        dataset_name = []

        output_ = 1
        for i, dataset_ in enumerate(self.datasets_data):
            if query == 'last':
                dataset_data = dataset_.dropna()
                class_labels = dataset_data.loc[:, dataset_data.columns.str.startswith("class")]
                row_tuples = dataset_data.drop(columns=list(class_labels.columns))

                last_row_np = np.asarray(row_tuples.tail(1), dtype=np.float64)
                last_class_label = np.asarray(class_labels.tail(1), dtype=np.float64)

                row_tuples = row_tuples.drop(row_tuples.index[len(row_tuples) - 1])
                class_labels = class_labels.drop(class_labels.index[len(class_labels) - 1])

                class_labels = np.asarray(class_labels, dtype=np.float64)
                tuples_np = np.asarray(row_tuples, dtype=np.float64)
                tuples_np = np.nan_to_num(tuples_np)

                if self.norm:
                    tuples_np = preprocessing.normalize(tuples_np)
                    last_row_np = preprocessing.normalize(last_row_np)

                data_set_lst.append(row_tuples)
                labels_lst.append(class_labels)

                last_row_data_lst.append(last_row_np)
                last_label_lst.append(last_class_label)
            elif query == 'all':
                dataset_data = dataset_.dropna()
                class_labels = dataset_data.loc[:, dataset_data.columns.str.startswith("class")]
                row_tuples = dataset_data.drop(columns=list(class_labels.columns))

                class_labels = np.asarray(class_labels, dtype=np.float64)
                tuples_np = np.asarray(row_tuples, dtype=np.float64)
                tuples_np = np.nan_to_num(tuples_np)

                if self.norm:
                    tuples_np = preprocessing.normalize(tuples_np)

                data_set_lst.append(dataset_data)
                labels_lst.append(class_labels)

                output_ = np.maximum(class_labels.shape[0], output_)
            elif query == 'None':
                dataset_data = dataset_.dropna()
                class_labels = dataset_data.loc[:, dataset_data.columns.str.startswith("class")]
                row_tuples = dataset_data.drop(columns=list(class_labels.columns))

                class_labels = np.asarray(class_labels, dtype=np.float64)
                tuples_np = np.asarray(row_tuples, dtype=np.float64)
                tuples_np = np.nan_to_num(tuples_np)
                if self.norm:
                    tuples_np = preprocessing.normalize(tuples_np)

                data_set_lst.append(dataset_data)
            dataset_name.append(self.get_dataset_name(i))
        if concat:
            ret_array_np = np.concatenate(data_set_lst, axis=0)
            ret_array_labels_np = np.concatenate(labels_lst, axis=0)
        else:
            ret_array_np = data_set_lst
            ret_array_labels_np = labels_lst

        if len(last_row_data_lst) >= 1:
            ret_lr_array_np = np.concatenate(last_row_data_lst, axis=0)
            ret_lr_array_labels_np = np.concatenate(last_label_lst, axis=0)
        else:
            ret_lr_array_np = None
            ret_lr_array_labels_np = None

        return ret_array_np, ret_array_labels_np, ret_lr_array_np, ret_lr_array_labels_np, dataset_name, output_

    def get_all_dataset_data_sr(self, query='last', label_files=None):
        vectors_list = []
        labels_list = []
        idx = 0
        output_ = 11111111
        # print(output_)
        for data_, vector_ in zip(self.datasets_data, self.vectors):
            dataset_name = self.get_dataset_name(idx=idx)

            if label_files is not None:
                label_ = label_files.loc[label_files['Dataset Name'] == dataset_name]['Y']
                labels_list.append(label_)
            else:
                if query == 'last':
                    dataset_data = data_.dropna()
                    class_labels = dataset_data.loc[:, dataset_data.columns.str.startswith("class")]
                    last_class_label = class_labels.tail(1)
                    labels_list.append(last_class_label)
                elif query == 'all':
                    dataset_data = data_.dropna()
                    class_labels = dataset_data.loc[:, dataset_data.columns.str.startswith("class")]
                    output_ = np.minimum(class_labels.shape[0], output_)
                    labels_list.append(class_labels)

            vectors_list.append(vector_)
            idx += 1

        if query == 'all':
            for i, labels_ in enumerate(labels_list):
                if labels_.shape[0] > output_:
                    for j in range(np.abs(labels_.shape[0] - output_)):
                        labels_list[i] = labels_list[i].drop(labels_list[i].index[len(labels_list[i]) - 1])
                        # labels_list[i] = np.asarray(labels_list[i]).reshape(-1, 1)

        vec_np = np.asarray(vectors_list)
        label_np = np.asarray(labels_list)
        if query == 'last':
            label_np = label_np.reshape(-1, 1)
        elif query == 'all':
            label_np = label_np.reshape(label_np.shape[0], label_np.shape[1])
        # elif query == 'all':
        #     label_np = label_np.reshape(-1, 1, 1)
        print(label_np.shape)
        return vec_np, label_np

    def get_all_dataset_data(self):
        if self.create_operator:
            dataset_data = self.datasets_data.dropna()
            class_labels = dataset_data.loc[:, self.datasets_data.columns.str.startswith("class")]
            row_tuples = dataset_data.drop(columns=list(class_labels.columns))
        else:
            df = pd.DataFrame()
            for df_data in self.datasets_data:
                df = pd.concat([df, df_data])
            df = df.dropna()
            class_labels = df.loc[:, df.columns.str.contains("class")]
            row_tuples = df.drop(columns=list(class_labels.columns))

        class_labels = np.asarray(class_labels, dtype=np.float64)

        tuples_np = np.asarray(row_tuples, dtype=np.float64)
        tuples_np = np.nan_to_num(tuples_np)

        if self.norm:
            tuples_np = preprocessing.normalize(tuples_np)

        if self.ret_class:
            return tuples_np, class_labels
        else:
            return tuples_np

    def __getitem__(self, idx):

        if self.ret_all_dataset:
            row_tuples = self.datasets_data[idx]

            class_labels = row_tuples.loc[:, row_tuples.columns.str.contains("class")]
            if not self.ret_label:
                row_tuples = row_tuples.drop(columns=list(class_labels.columns))

            dataset_id = self.current_dataset_id

            if row_tuples.shape[0] < self.k_dim:
                zero_pad = np.zeros(shape=(self.k_dim, row_tuples.shape[1]))
                zero_pad[:row_tuples.shape[0], :] = row_tuples
                row_tuples = zero_pad

            self.current_dataset_id += 1
        elif self.ret_table:
            dataset_data = self.datasets_data[self.current_dataset_id]
            row_tuples = dataset_data[self.data_idx]

            class_labels = row_tuples.loc[:, row_tuples.columns.str.contains("class")]
            if not self.ret_label:
                row_tuples = row_tuples.drop(columns=list(class_labels.columns))

            dataset_id = np.zeros(shape=(self.k_dim, 1))
            dataset_id[:] = self.current_dataset_id

            self.data_idx = self.data_idx + 1
            if self.data_idx >= len(dataset_data):
                self.data_idx = 0
                self.current_dataset_id = self.current_dataset_id + 1
                if self.current_dataset_id >= len(self.datasets_data):
                    self.current_dataset_id = 0

            if row_tuples.shape[0] < self.k_dim:
                zero_pad = np.zeros(shape=(self.k_dim, row_tuples.shape[1]))
                zero_pad[:row_tuples.shape[0], :] = row_tuples
                row_tuples = zero_pad
        else:
            dataset_data = self.datasets_data[self.current_dataset_id]
            row_tuples = dataset_data.iloc[[idx]]
            class_labels = row_tuples.filter(regex='class')
            row_tuples = row_tuples.drop(columns=list(class_labels.columns))

        tuples_np = np.asarray(row_tuples, dtype=np.float64)
        tuples_np = np.nan_to_num(tuples_np)

        if self.norm:
            if self.ret_table:
                tuples_np = preprocessing.normalize(tuples_np)
            else:
                tuples_np = preprocessing.normalize([tuples_np])[0]

        if self.ret_table:
            tuples_np = tuples_np.flatten()

        # class_labels.values.flatten().tolist()
        # class_tuple_np = np.asarray(class_labels, dtype=np.float64)

        # if class_labels.shape[0] < self.k_dim:
        #     zero_pad = np.zeros(shape=(self.k_dim, class_tuple_np.shape[1]))
        #     zero_pad[:class_labels.shape[0], :] = class_tuple_np
        #     zero_pad[class_labels.shape[0]:, :] = np.nan
        #     class_tuple_np = zero_pad
        #     dataset_id[class_labels.shape[0]:] = -99

        return tuples_np, dataset_id

    def __len__(self):
        return self.data_len


class Pipeline_Dataset:

    def __init__(self, data_path, norm=True, ret_table=True, ret_label=True, k_dim=50, ret_all_dataset=True,
                 create_operator=False, ret_class=False, Vectors=None):
        self.datasets_path = data_path
        self.norm = norm

        self.ret_table = ret_table
        self.k_dim = k_dim

        self.ret_label = ret_label
        self.ret_class = ret_class
        self.create_operator = create_operator

        self.vectors = Vectors

        self.ret_all_dataset = ret_all_dataset

        check_path(path=self.datasets_path)
        if type(self.datasets_path) is list:
            self.datasets_path = data_path
        elif os.path.isdir(self.datasets_path):
            files_ = os.listdir(self.datasets_path)
            paths_ = []
            for file in files_:
                filepath_ = os.path.join(self.datasets_path, file)
                paths_.append(filepath_)
            self.datasets_path = paths_

    def get_Dataloader(self):
        return DataBuilder_Pipeline(datasets_filepaths=self.datasets_path,
                                    ret_Table=self.ret_table,
                                    ret_label=self.ret_label,
                                    k_dim=self.k_dim,
                                    norm=self.norm,
                                    create_operator=self.create_operator,
                                    ret_class=self.ret_class,
                                    Vectors=self.vectors)


class Dataset:

    def __init__(self, data_conf, return_labels=True, split=True,
                 train=0.75, test=0.15, val=0.1, norm=True, augment=True,
                 ret_table=True, k_dim=5, ret_all_dataset=True):

        self.datasets_dict = data_conf
        self.categorical_columns = []
        self.normalise_data = norm
        self.augmentation = augment
        self.ret_all_dataset = ret_all_dataset
        self.ret_tables = True
        self.return_labels = True
        self.ret_table = ret_table
        if self.ret_all_dataset:
            self.k_dim = 0
        else:
            self.k_dim = k_dim

        # self.add_categorical_columns()
        self.split = split
        for dataset_name, dataset in self.datasets_dict.items():
            self.set_paths(dataset_name=dataset_name, filepath=dataset['path'])
        if split:
            self.set_train_test_idx(test=test,
                                    train=train,
                                    val=val)

    def set_paths(self, filepath, dataset_name):
        if os.path.isfile(filepath):
            return
        else:
            self.datasets_dict[dataset_name]['dir_path'] = filepath
            files_ = os.listdir(filepath)
            paths_ = []
            for file in files_:
                filepath_ = os.path.join(filepath, file)
                paths_.append(filepath_)
            self.datasets_dict[dataset_name]['path'] = paths_

    def set_train_test_idx(self, train, test, val):
        for name, dataset in self.datasets_dict.items():
            dataset_path = dataset['dir_path']
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
                if self.ret_all_dataset:
                    files_to_idx = {i: os.path.join(dataset_path, path_) for i, path_ in enumerate(files_)}
                    train_idx, test_idx = train_test_split(files_to_idx, train_size=train, test_size=(test + val))
                    if val > 0:
                        test_idx, val_idx = train_test_split(test_idx, test_size=val)
                        val_idx_lst = val_idx

                    test_idx_lst = test_idx
                    train_idx_lst = train_idx

                    for file in files_:
                        df = pd.read_csv(os.path.join(dataset_path, file))
                        if self.k_dim < df.shape[0]:
                            self.k_dim = df.shape[0]
                else:
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

    def get_Dataset_dataloader(self, dataset_name, all_datasets=True):
        if all_datasets:
            dataset_path, tuples_columns_train, tuples_columns_test, tuples_columns_val, class_columns = [], [], [], [], []
            class_cols_train, class_cols_test, class_cols_val = [], [], []
            train_idx, test_idx, val_idx = [], [], []
            for dataset_name, dict_ in self.datasets_dict.items():
                dataset_path.extend(dict_['path'])
                class_columns.extend(dict_['class'])

                train_idx.extend(dict_['train_idx'])
                test_idx.extend(dict_['test_idx'])
                val_idx.extend(dict_['val_idx'])

                # For collumns tuples
                tuples_columns_train.extend([dict_['columns'] for i in dict_['train_idx']])
                tuples_columns_test.extend([dict_['columns'] for i in dict_['test_idx']])
                tuples_columns_val.extend([dict_['columns'] for i in dict_['val_idx']])

                # For Labels
                class_cols_train.extend([dict_['class'] for i in dict_['train_idx']])
                class_cols_test.extend([dict_['class'] for i in dict_['test_idx']])
                class_cols_val.extend([dict_['class'] for i in dict_['val_idx']])

            train_data_builder = DataBuilder(filepath=dataset_path,
                                             dir_path=self.datasets_dict[dataset_name]['dir_path'],
                                             columns=tuples_columns_train,
                                             labels=class_cols_train,
                                             ret_labels=True,
                                             col_idx=train_idx,
                                             norm=self.normalise_data,
                                             ret_table=self.ret_table,
                                             k_dim=self.k_dim,
                                             ret_all_dataset=self.ret_all_dataset)

            test_data_builder = DataBuilder(filepath=dataset_path,
                                            dir_path=self.datasets_dict[dataset_name]['dir_path'],
                                            columns=tuples_columns_test,
                                            labels=class_cols_test,
                                            ret_labels=True,
                                            col_idx=test_idx,
                                            norm=self.normalise_data,
                                            ret_table=self.ret_table,
                                            k_dim=self.k_dim,
                                            ret_all_dataset=self.ret_all_dataset)

            val_data_builder = DataBuilder(filepath=dataset_path,
                                           dir_path=self.datasets_dict[dataset_name]['dir_path'],
                                           columns=tuples_columns_val,
                                           labels=class_cols_val,
                                           ret_labels=True,
                                           col_idx=val_idx,
                                           norm=self.normalise_data,
                                           ret_table=self.ret_table,
                                           k_dim=self.k_dim,
                                           ret_all_dataset=self.ret_all_dataset)
        else:
            print('in')
            dataset_path = self.datasets_dict[dataset_name]['path']
            tuples_columns = self.datasets_dict[dataset_name]['columns']
            class_columns = self.datasets_dict[dataset_name]['class']

            train_idx = self.datasets_dict[dataset_name]['train_idx']
            test_idx = self.datasets_dict[dataset_name]['test_idx']
            val_idx = self.datasets_dict[dataset_name]['val_idx']

            train_data_builder = DataBuilder(filepath=dataset_path,
                                             ret_labels=True,
                                             dir_path=None,
                                             columns=tuples_columns,
                                             labels=class_columns,
                                             col_idx=train_idx,
                                             norm=self.normalise_data,
                                             ret_table=self.ret_table,
                                             k_dim=self.k_dim,
                                             ret_all_dataset=self.ret_all_dataset)

            test_data_builder = DataBuilder(filepath=dataset_path,
                                            ret_labels=True,
                                            dir_path=None,
                                            columns=tuples_columns,
                                            labels=class_columns,
                                            col_idx=test_idx,
                                            norm=self.normalise_data,
                                            ret_table=self.ret_table,
                                            k_dim=self.k_dim,
                                            ret_all_dataset=self.ret_all_dataset)

            val_data_builder = DataBuilder(filepath=dataset_path,
                                           ret_labels=True,
                                           dir_path=None,
                                           columns=tuples_columns,
                                           labels=class_columns,
                                           col_idx=test_idx,
                                           norm=self.normalise_data,
                                           ret_table=self.ret_table,
                                           k_dim=self.k_dim,
                                           ret_all_dataset=self.ret_all_dataset)

        return train_data_builder, test_data_builder, val_data_builder


def load_data(dataset_config, ):
    if not os.path.exists(dataset_config):
        AssertionError(f'Filepath {dataset_config} is not exist')

    full_path_config = os.path.join(os.getcwd(), dataset_config)
    # path_config = os.path.abspath(dataset_config)

    with open(full_path_config, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    return data_loaded
