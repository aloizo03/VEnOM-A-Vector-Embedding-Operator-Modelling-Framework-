import numpy as np
import pandas as pd

from data.Dataset import Pipeline_Dataset
from utils_.utils import create_operator, create_ML_model, predict_dbscan, predict_operator, predict_linear_regression_operator, fit_operator, predict_time_series_model, compute_rank, compute_sum, compute_avg, compute_eig_value
from torch.utils.data import DataLoader
from models.torch_utils import select_device
import csv

from models.vectorEmbedding import Num2Vec
import torch
import pickle
import logging
import os
import time
import random 
import operator as op

logger = logging.getLogger(__name__)

class Train_Operator:
    def __init__(self, vec_input_path, pred_path, out_path, sr, model_path):
        
        self.device = select_device()
        ckpt = torch.load(model_path)

        model_info = ckpt['model_info']
        input_dim = model_info['d_numerical']
        d_out = model_info['d_out']
        d_tokens = model_info['d_token']
        num_layers = model_info['num_layers']
        head_num = model_info['head_num']
        attention_dropout = model_info['att_dropout']
        ffn_dropout = model_info['ffn_dropout']
        ffn_dim = model_info['ffn_dim']
        activation = model_info['activation']
        dim_k = model_info['dim_k']
        table_input = model_info['ret_table']

        self.model = Num2Vec(d_numerical=input_dim,
                             d_out=d_out,  # create a 3d representation
                             n_layers=num_layers,
                             d_token=d_tokens,
                             head_num=head_num,
                             attention_dropout=attention_dropout,
                             ffn_dropout=ffn_dropout,
                             ffn_dim=ffn_dim,
                             activation=activation,
                             train=False,
                             input_table=table_input,
                             d_k=dim_k)

        self.model.load_state_dict(ckpt['model_state_dict'])
        
        self.vec_input_path = vec_input_path
        self.pred_path = pred_path
        self.dataset = Pipeline_Dataset(data_path=self.pred_path,
                                        norm=True,
                                        ret_table=table_input,
                                        k_dim=dim_k)
        self.out_path = out_path

        self.sr = sr

        self.vectors = self.read_vectors()
        logging.basicConfig(filename=os.path.join(self.out_path, 'log_file.log'), encoding='utf-8',
                            level=logging.DEBUG)
        logger.setLevel(logging.INFO)

    def read_vectors(self):
        with open(self.vec_input_path, 'rb') as handle:
            b = pickle.load(handle)
        return b

    def select_datasets_uniform(self):
        filenames_list = list(self.vectors.keys())
        random.shuffle(filenames_list)

        random_choice_datasets = np.random.choice(filenames_list, int(np.floor(len(filenames_list) * self.sr)))
        random_choice_datasets = list(random_choice_datasets)

        selected_vectors = [self.vectors[dataset_] for dataset_ in random_choice_datasets]

        return random_choice_datasets, selected_vectors
    
    def create_vectors_(self):
        DataBuilder = self.dataset.get_Dataloader()
        dataloader = DataLoader(DataBuilder, batch_size=1, shuffle=False)

        vectors_output = []
        vectors_dataset_idx = []

        with torch.no_grad():
            for batch_idx, batch_ in enumerate(dataloader):
                features_, dataset_idx = batch_
                features_ = features_.to(self.device)
                _, vectors, _ = self.model(features_)
                    
                vectors_output.append(vectors.detach().cpu().numpy())
                vectors_dataset_idx.append(dataset_idx.numpy())

        vectors_out_np = np.concatenate(vectors_output, axis=0)
        vectors_dataset_idx_np = np.concatenate(vectors_dataset_idx, axis=0)
        
        self.dataset =  Pipeline_Dataset(self.pred_path, norm=True, create_operator=False, ret_class=True, ret_all_dataset=True, Vectors=vectors_out_np)

        return vectors_out_np, vectors_dataset_idx_np
    
    def read_scores(self, score_file):
        if os.path.isfile(score_file):
            return pd.read_csv(score_file)
        else:
            None

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

    def train_operator(self, ML_model_name, query, op_name, scores_file=None, repititions=1):
        logger.info(f'Experiments for operator {op_name}')
        start_time = time.time()
        pred_vectors, pred_dataset_idx = self.create_vectors_()
        
        selected_datasets, selected_vectors = self.select_datasets_uniform()

        dataset_train = Pipeline_Dataset(selected_datasets, norm=True, create_operator=False, ret_class=True, ret_all_dataset=True, Vectors=selected_vectors)

        if scores_file is not None:
            score_file_df = self.read_scores(scores_file)
        else:
            score_file_df = None

        data_builder_train = dataset_train.get_Dataloader()
        X_train, y_train = data_builder_train.get_all_dataset_data_sr(query=query, label_files=score_file_df)
        data_test = self.dataset.get_Dataloader()
        X_test, y_test = data_test.get_all_dataset_data_sr(query=query, label_files=None)

        if score_file_df is not None:
                if op_name.lower() == 'rank' or op_name.lower() == 'sum' or op_name.lower() == 'avg' or op_name.lower() == 'eig':
                    X_test, y_test, x_test, y_test, dataset_names, output_dim = data_test.get_all_data_per_dataset(query=query, concat=False)
                    y_test = []
                    for X_ in X_test:
                        X_ = np.asarray(X_)
                        if op_name.lower() == 'rank':
                            y_pred = compute_rank(X=X_)
                        elif op_name.lower() == 'sum':
                            y_pred = compute_sum(X=X_)
                        elif op_name.lower() == 'avg':
                            y_pred = compute_avg(X=X_)
                        elif op_name.lower() == 'eig':
                            y_pred = compute_eig_value(X=X_)
                        y_test.append(y_pred)
                    y_test = np.asarray(y_test)
        metrics_results_average = {
            'Acc': 0, 'r^2': 0, 'NRMSE': 0, 'RMSE':0 , 'MAE':0, 'MAD': 0, 'MaPE': 0, 'Exec time': 0
        }
        exec_time = time.time() - start_time
        logger.info(f'Before the ml creation exec time: {exec_time}')
        for i in range(repititions):
            start_time = time.time()
            ML_model = create_ML_model(X=X_train, y=y_train, name=ML_model_name)

            # for j in range(pred_vectors.shape[0]):
            if ML_model_name == 'perceptron':
                acc, r_2_score, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss = predict_operator(operator=ML_model, X=pred_vectors, y=y_test)
                exec_time = time.time() - start_time
                logger.info(f'Execution time : {exec_time}')

                metrics_results_average['NRMSE'] += nrmse_loss
                metrics_results_average['RMSE'] += rmse_loss
                metrics_results_average['MAE'] += mae_loss
                metrics_results_average['MAD'] += mad_loss
                metrics_results_average['MaPE'] += MaPE_loss
                metrics_results_average['Exec time'] += exec_time
                metrics_results_average['Acc'] += acc
            elif ML_model_name == 'perceptron_regression':
                exec_time = time.time() - start_time
                logger.info(f'Execution time : {exec_time}')
                r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss = predict_linear_regression_operator(operator=ML_model, X=pred_vectors, y=y_test)
                metrics_results_average['r^2'] += r2
                metrics_results_average['NRMSE'] += nrmse_loss
                metrics_results_average['RMSE'] += rmse_loss
                metrics_results_average['MAE'] += mae_loss
                metrics_results_average['MAD'] += mad_loss
                metrics_results_average['MaPE'] += MaPE_loss
                metrics_results_average['Exec time'] += exec_time
        
        metrics_results_average['NRMSE'] /= repititions
        metrics_results_average['RMSE'] /= repititions
        metrics_results_average['MAE'] /= repititions
        metrics_results_average['MAD'] /= repititions
        metrics_results_average['MaPE'] /= repititions
        metrics_results_average['Exec time'] /= repititions
        metrics_results_average['Acc'] /= repititions
        metrics_results_average['r^2'] /= repititions

        
        self.save_csv_file(header=['Operator Name', 'Data Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time'],
                                            record=[op_name, f'Average All with Query: {query}', metrics_results_average['Acc'], '-', metrics_results_average['NRMSE'], metrics_results_average['RMSE'], metrics_results_average['MAE'], metrics_results_average['MAD'], metrics_results_average['MaPE'], metrics_results_average['r^2']],
                                            filename='results_out.csv')


