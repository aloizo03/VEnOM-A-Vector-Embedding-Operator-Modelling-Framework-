from h11 import Data
import numpy as np

from data.Dataset import Pipeline_Dataset, graph_Pipeline_Dataset
from utils_.utils import create_operator, create_graph_operator, create_ML_model, predict_dbscan, predict_operator, predict_linear_regression_operator, fit_operator, predict_time_series_model, compute_rank, compute_sum, compute_avg, compute_eig_value
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
from utils_.utils import calculate_cosine_similarity

import pickle
import logging
import os
import time
import scipy
import random 
import operator as op
import pandas as pd
import csv

logger = logging.getLogger(__name__)

class EXP_Accuracy:
    def __init__(self, input_data_dir, out_path_dir, operator_name):
        
        self.input_path = input_data_dir
        self.out_path = out_path_dir
        self.operator_name = operator_name

        logging.basicConfig(filename=os.path.join(self.out_path, 'log_file.log'), encoding='utf-8',
                            level=logging.DEBUG)
        logger.setLevel(logging.INFO)
    
    def save_predictions(self, y_pred, dataset_names, out_name='scores.csv'):
        if y_pred.shape == 1:
            df = pd.DataFrame({'Dataset Name': dataset_names,
                            'Y': y_pred})
        else:
            df = pd.DataFrame({'Dataset Name': dataset_names,
                            'Y': y_pred.reshape(-1)})
        
        df.to_csv(os.path.join(self.out_path, out_name), index=False)

    def create_op(self, query='last', return_labels=False, labels_name='scores.csv'):

        
        start_time = time.time()

        if self.operator_name.lower() == 'rank' or self.operator_name.lower() == 'sum' or self.operator_name.lower() == 'avg' or self.operator_name.lower() == 'eig':
            dataset = Pipeline_Dataset(self.input_path, norm=True, create_operator=False, ret_class=True, ret_all_dataset=True)
            data_builder_train = dataset.get_Dataloader()

            X_train, y_train, x_test, y_test, dataset_names, output_dim = data_builder_train.get_all_data_per_dataset(query=query, concat=False)
            predictions = []
            for X_ in X_train:
                X_ = np.asarray(X_)
                if self.operator_name.lower() == 'rank':
                    y_pred = compute_rank(X=X_)
                elif self.operator_name.lower() == 'sum':
                    y_pred = compute_sum(X=X_)
                elif self.operator_name.lower() == 'avg':
                    y_pred = compute_avg(X=X_)
                elif self.operator_name.lower() == 'eig':
                    y_pred = compute_eig_value(X=X_)

                predictions.append(y_pred)
            exec_time = time.time() - start_time
            logger.info(f'Operator {self.operator_name} has been created exec time : {exec_time}')
            if not return_labels:
                self.save_predictions(predictions, dataset_names)
            else:
                df = pd.DataFrame({'Dataset Name': dataset_names,
                           'Y': predictions})
                return df
        elif self.operator_name.lower() == 'bc' or self.operator_name.lower() == 'ebc' or self.operator_name.lower() == 'cc' or self.operator_name.lower() == 'ec' or self.operator_name.lower() == 'pr':
            
            dataset = graph_Pipeline_Dataset(data_path=self.input_path, return_label=False)
            DataBuilder = dataset.get_Dataloader()
            graphs = DataBuilder.get_all_data()
            
            dataset_names = DataBuilder.datasets_paths
            y = create_graph_operator(self.operator_name, graph_list=dataset_names)
            y_np = np.asarray(y, dtype=np.float64)
            if not return_labels:
                self.save_predictions(y_np, dataset_names)
            else:
                df = pd.DataFrame({'Dataset Name': dataset_names, 'Y': y_np})
                return df
        else:
            dataset = Pipeline_Dataset(self.input_path, norm=True, create_operator=False, ret_class=True, ret_all_dataset=True)
            data_builder_train = dataset.get_Dataloader()

            X_train, y_train, x_test, y_test, dataset_names, output_dim = data_builder_train.get_all_data_per_dataset(query=query, concat=True)
            operator_ = create_operator(X=X_train, 
                                        y=y_train,
                                        name=self.operator_name)
            # self.save_predictions(y_test, dataset_names, out_name='labels.csv')
            
            if x_test is not None and y_test is not None:
                if self.operator_name.lower() == 'linear_regression':
                    r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, y_pred = predict_linear_regression_operator(X=x_test, y=y_test, operator=operator_, ret_preds=True)
                    exec_time = time.time() - start_time
                    logger.info(f'Operator {self.operator_name} has been created exec time : {exec_time}')
 
                    if not return_labels:
                        self.save_predictions(y_pred, dataset_names)
                    else:
                        df = pd.DataFrame({'Dataset Name': dataset_names, 'Y': y_pred})
                        return df
                    
                elif op.contains(self.operator_name.lower(), 'arima') or op.contains(self.operator_name.lower(), 'holt_winter'):
                    r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, y_pred = predict_time_series_model(X=x_test, y=y_test, operator=operator_, operator_name=self.operator_name, ret_preds=True)
                    
                    exec_time = time.time() - start_time
                    logger.info(f'Operator {self.operator_name} has been created exec time : {exec_time}')
 
                    if not return_labels:
                        self.save_predictions(y_pred, dataset_names)
                    else:
                        df = pd.DataFrame({'Dataset Name': dataset_names, 'Y': y_pred})
                        return df
                else:
                    acc, r_2_score, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, y_pred = predict_operator(X=x_test, y=y_test, operator=operator_, ret_preds=True)
                    
                    exec_time = time.time() - start_time
                    logger.info(f'Operator {self.operator_name} has been created exec time : {exec_time}')
 
                    if not return_labels:
                        self.save_predictions(y_pred, dataset_names)
                    else:
                        df = pd.DataFrame({'Dataset Name': dataset_names, 'Y': y_pred})
                        return df
                    
                    
                
