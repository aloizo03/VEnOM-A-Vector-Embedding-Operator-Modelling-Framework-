import numpy as np

from data.Dataset import Pipeline_Dataset
from utils_.utils import create_operator, create_ML_model, predict_dbscan, predict_operator, predict_linear_regression_operator, fit_operator, predict_time_series_model
from sklearn.model_selection import train_test_split

import pickle
import logging
import os
import time
import scipy
import random 
import operator as op
import csv

logger = logging.getLogger(__name__)


class SR_Operator_Creation:

    def __init__(self, vec_input_path, out_path, sr):
        self.vec_input_path = vec_input_path
        self.out_path = out_path
        self.sr = sr

        logging.basicConfig(filename=os.path.join(self.out_path, 'log_file.log'), encoding='utf-8',
                            level=logging.DEBUG)
        logger.setLevel(logging.INFO)

        self.vectors = self.read_vectors()

        logger.info('Read Vectors')

    def read_vectors(self):
        with open(self.vec_input_path, 'rb') as handle:
            b = pickle.load(handle)
        return b

    def select_datasets(self):
        filenames_list = list(self.vectors.keys())

        random_choice_datasets = np.random.choice(filenames_list, int(np.floor(len(filenames_list) * self.sr)))
        random_choice_datasets = list(random_choice_datasets)
        return random_choice_datasets

    
    def select_datasets_uniform(self):
        filenames_list = list(self.vectors.keys())
        random.shuffle(filenames_list)

        random_choice_datasets = np.random.choice(filenames_list, int(np.floor(len(filenames_list) * self.sr)))
        random_choice_datasets = list(random_choice_datasets)

        selected_vectors = [self.vectors[dataset_] for dataset_ in random_choice_datasets]

        return random_choice_datasets, selected_vectors
    
    def save_operator(self, filename, operator):
        file_name = os.path.join(self.out_path, f'operator_{filename}_{time.time()}.pkl')
        with open(file_name, 'wb') as f:
            pickle.dump(operator, f)

    def save_csv_file(self, header, record, filename='results_out.csv'):
        outfile = os.path.join(self.out_path, filename)
        if os.path.isfile(outfile):
            csv_file = open(outfile,'a')
            writer = csv.writer(csv_file,delimiter=',')
            writer.writerow([r for r in record])
        else:
            with open(outfile,'w') as csvfile:    
                writer = csv.writer(csvfile, delimiter=',')
                # Gives the header name row into csv
                writer.writerow([g for g in header])  
                writer.writerow([r for r in record]) 
        logger.info('Writing the results into a csv file')

    def create_operator(self, operator_name, split_data=False):
        start_time = time.time()
        selected_files, selected_vectors = self.select_datasets_uniform()

        dataset_train = Pipeline_Dataset(selected_files, norm=True, create_operator=False, ret_class=True, ret_all_dataset=True, Vectors=selected_vectors)
        data_builder_train = dataset_train.get_Dataloader()

        X_train, y_train, dict_ = data_builder_train.get_all_dataset_data_sr(query='all')

        start_time = time.time()
        operator_ = create_operator(name=operator_name, X=X_train, y=y_train)
        exec_time = time.time() - start_time
        logger.info(f'Operator {operator_name} has been created exec time : {exec_time}')
        self.save_operator(filename=operator_name, operator=operator_)
