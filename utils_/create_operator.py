from data.Dataset import Pipeline_Dataset, graph_Pipeline_Dataset, Image_Pipeline_Dataset
from utils_.utils import get_metrics, create_operator, create_graph_operator, predict_operator, predict_linear_regression_operator, predict_time_series_model, predict_dbscan, create_ML_model, normalize_label
import pickle
import logging
import os
import time
import operator as op
import csv
import numpy as np
import pandas as pd 

logger = logging.getLogger(__name__)


class Create_Operator:

    def __init__(self, out_path, operator_dir):
        self.out_path = out_path

        self.operator_dir = operator_dir

        self.operator = self.read_operator()
        logging.basicConfig(filename=os.path.join(self.out_path, 'log_file.log'), encoding='utf-8',
                            level=logging.DEBUG)
        logger.setLevel(logging.INFO)

    def read_operator(self):
        if self.operator_dir is None:
            return None
        with open(self.operator_dir, 'rb') as f:
            operator_ = pickle.load(f)
        return operator_

    def save_csv_file(self, header, record, filename='results_out_new.csv'):
        outfile = os.path.join(self.out_path, filename)
        if os.path.isfile(outfile):
            csv_file = open(outfile,'a')
            writer = csv.writer(csv_file,delimiter=',')
            writer.writerow([r for r in record])
        else:
            with open(outfile,'w') as csvfile:    
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow([g for g in header])  
                writer.writerow([r for r in record]) 
        logger.info('Writing the results into a csv file')

    def load_dict(self, path_):
        with open(path_, 'rb') as handle:
            b = pickle.load(handle)
        return b

    def save_model(self, filename, operator):
        file_name = os.path.join(self.out_path, f'operator_{filename}_{time.time()}.pkl')
        with open(file_name, 'wb') as f:
            pickle.dump(operator, f)

    
    def create_operator(self, operator_name, most_relevant_data_path, target_labels=None, repetitions=5, use_vectors=True, load_rel_data=True, return_res=False, labels_dict=None, data_type=1):
        # TODO: Add the metrics with all the labels and the predictions at the end of repetitions
        if load_rel_data:
            selected_dic = self.load_dict(path_=most_relevant_data_path)
        else:
            selected_dic = most_relevant_data_path
            
        metrics_results_average = {
            'Acc': 0, 'r^2': 0, 'NRMSE': 0, 'RMSE':0 , 'MAE':0, 'MAD': 0, 'MaPE': 0, 'Exec time': 0
        }

        if labels_dict is not None and isinstance(labels_dict, str):
            labels_path = labels_dict
            labels_dict = pd.read_csv(labels_dict)

        if target_labels is not None and isinstance(target_labels, str):
            target_labels_path = target_labels
            labels_dict_target = pd.read_csv(target_labels_path)
        
        count = 0
        most_relevant_data = selected_dic['selected_dataset']
        pred_vectors = selected_dic['Pred_Dataset']
        pred_y = []
        label_y = []
        
        for j in range(repetitions):
            for dataset_name, dataset_list in most_relevant_data.items():
                count += 1
                
                if data_type == 2:
                    #TODO: Make to show to give the labels of the test graph datasets
                    # y_test = create_graph_operator(name=operator_name.lower(), graph_list=[dataset_name])
                    dataset_pred_vectors = pred_vectors[dataset_name]
                    dataset_predict = graph_Pipeline_Dataset(data_path=dataset_name,
                                                             labels_dict=labels_dict_target,
                                                             Vectors=dataset_pred_vectors, 
                                                             return_label=True)
                    data_builder_test = dataset_predict.get_Dataloader()
                    X_test, y_test = data_builder_test.get_all_data()
                elif data_type == 3:
                    #TODO: Make to show to give the labels of the test image datasets
                    dataset_pred_vectors = pred_vectors[dataset_name]
                    dataset_predict = Image_Pipeline_Dataset(dataset_name,
                                                             create_op=True,
                                                             labels_dict=labels_dict_target,
                                                             labels_dict_path=target_labels_path,
                                                             Vectors=dataset_pred_vectors,
                                                             return_label=True)
                    data_builder_test = dataset_predict.get_Dataloader()
                    X_test, y_test = data_builder_test.get_data_op_modelling(filename=dataset_name)
                else:
                    dataset_predict = Pipeline_Dataset(dataset_name, norm=True, create_operator=True, ret_class=True)
                    data_builder_test = dataset_predict.get_Dataloader()
                    X_test, y_test = data_builder_test.get_all_dataset_data()

                if self.operator is not None:
                    start_time = time.time()
                    if op.contains(operator_name.lower(), 'regression'):
                        r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss = predict_linear_regression_operator(
                                operator=self.operator, X=X_test, y=y_test)
                        logger.info('Use operator F')
                        # self.save_model(filename=operator_name, operator=self.operator)
                        logger.info(
                                    f'Operator name: {operator_name}, for dataset {dataset_name} r2 score is: {r2}, NRMSE: {nrmse_loss}, RMSE: {rmse_loss}, MAE : {mae_loss}, MAD : {mad_loss}, MaPE : {MaPE_loss}')
                        exec_time = time.time() - start_time
                        logger.info(f'Execution time : {exec_time}')
                        
                        if not return_res:
                            self.save_csv_file(header=['Operator Name', 'Data Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time'],
                                                record=[operator_name, dataset_name, '-', r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, exec_time],
                                                filename='results_out.csv')
                        metrics_results_average['r^2'] += r2
                        metrics_results_average['NRMSE'] += nrmse_loss
                        metrics_results_average['RMSE'] += rmse_loss
                        metrics_results_average['MAE'] += mae_loss
                        metrics_results_average['MAD'] += mad_loss
                        metrics_results_average['MaPE'] += MaPE_loss
                        metrics_results_average['Exec time'] += exec_time
                        
                    elif op.contains(operator_name.lower(), 'arima') or op.contains(operator_name.lower(), 'holt_winter'):
                        r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss = predict_time_series_model(
                                operator=self.operator, X=X_test, y=y_test, operator_name=operator_name)
                        logger.info('Use operator F')

                        logger.info(
                                    f'Operator name: {operator_name}, for dataset {dataset_name} r2 score is: {r2}, NRMSE: {nrmse_loss}, RMSE: {rmse_loss}, MAE : {mae_loss}, MAD : {mad_loss}, MaPE : {MaPE_loss}')
                        exec_time = time.time() - start_time
                        logger.info(f'Execution time : {exec_time}')
                        
                        if not return_res:
                            self.save_csv_file(header=['Operator Name', 'Data Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time'],
                                                record=[operator_name, dataset_name, '-', r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, exec_time],
                                                filename='results_out.csv')
                        metrics_results_average['r^2'] += r2
                        metrics_results_average['NRMSE'] += nrmse_loss
                        metrics_results_average['RMSE'] += rmse_loss
                        metrics_results_average['MAE'] += mae_loss
                        metrics_results_average['MAD'] += mad_loss
                        metrics_results_average['MaPE'] += MaPE_loss
                        metrics_results_average['Exec time'] += exec_time
                    elif op.contains(operator_name.lower(), 'dbscan'):
                        acc, r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss = predict_dbscan(
                        operator=operator_, X=X_test, y=y_test)
                        logger.info(f'Execution time : {exec_time}')
                        
                        if not return_res:
                            self.save_csv_file(header=['Operator Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time'],
                                            record=[operator_name, acc, '-', nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, exec_time],
                                            filename='results_out.csv')
                        metrics_results_average['r^2'] += r2
                        metrics_results_average['NRMSE'] += nrmse_loss
                        metrics_results_average['RMSE'] += rmse_loss
                        metrics_results_average['MAE'] += mae_loss
                        metrics_results_average['MAD'] += mad_loss
                        metrics_results_average['MaPE'] += MaPE_loss
                        metrics_results_average['Exec time'] += exec_time
                    else:
                        acc, r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss = predict_operator(operator=self.operator, X=X_test, y=y_test)
                        logger.info('Use operator F, above threshold')
                        # self.save_model(filename=operator_name, operator=self.operator)
                        logger.info(f'Operator name: {operator_name}, for dataset {dataset_name} accuracy is: {acc}, r^2: {r2}, NRMSE: {nrmse_loss}, RMSE: {rmse_loss}, MAE : {mae_loss}, MAD : {mad_loss}, MaPE : {MaPE_loss}')
                        exec_time = time.time() - start_time
                        logger.info(f'Execution time : {exec_time}')
                        if not return_res:
                            self.save_csv_file(header=['Operator Name', 'Data Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time'],
                                                record=[operator_name, dataset_name, acc, '-', nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, exec_time],
                                                filename='results_out.csv')
                        metrics_results_average['r^2'] += r2
                        metrics_results_average['NRMSE'] += nrmse_loss
                        metrics_results_average['RMSE'] += rmse_loss
                        metrics_results_average['MAE'] += mae_loss
                        metrics_results_average['MAD'] += mad_loss
                        metrics_results_average['MaPE'] += MaPE_loss
                        metrics_results_average['Exec time'] += exec_time
                        metrics_results_average['Acc'] += acc
                else:
                    dataset_filenames = dataset_list[0]
                    dataset_vectors = dataset_list[1]
                    
                    dataset_pred_vectors = pred_vectors[dataset_name]
                    
                    start_time = time.time()

                    if use_vectors:
                        if data_type == 1:
                            dataset_train = Pipeline_Dataset(dataset_filenames, norm=True, create_operator=False, ret_class=True, Vectors=dataset_vectors)
                            data_builder_train = dataset_train.get_Dataloader()
                            X_train, y_train = data_builder_train.get_all_dataset_data_sr(query='last', label_files=labels_dict)
                            dataset_predict = Pipeline_Dataset(dataset_name, norm=True, create_operator=False, ret_class=True, Vectors=dataset_pred_vectors)
                            data_builder_test = dataset_predict.get_Dataloader()
                            X_test, y_test = data_builder_test.get_all_dataset_data_sr(query='last', label_files=None)
                        elif data_type == 2:
                            dataset_train = graph_Pipeline_Dataset(data_path=dataset_filenames,
                                                             labels_dict=labels_dict,
                                                             Vectors=dataset_vectors, 
                                                             return_label=True)
                            data_builder_train = dataset_train.get_Dataloader()
                            X_train, y_train = data_builder_train.get_all_data()
                        elif data_type == 3:
                            dataset_pred_vectors = pred_vectors[dataset_name]
                            dataset_train = Image_Pipeline_Dataset(dataset_filenames,
                                                                        labels_dict=labels_dict,
                                                                        Vectors=dataset_vectors,
                                                                        create_op=True,
                                                                        return_label=True)
                            data_builder_train = dataset_train.get_Dataloader()
                            print('Start Getting training Data')
                            X_train, y_train = data_builder_train.get_all_data_operator_modelling(dataset_filenames)
                            print('Finish Getting training Data')
                            # dataset_pred_vectors = pred_vectors[dataset_name]
                            # dataset_predict = Image_Pipeline_Dataset(dataset_name,
                            #                                  create_op=True,
                            #                                  labels_dict=labels_dict_target,
                            #                                  labels_dict_path=target_labels_path,
                            #                                  Vectors=dataset_pred_vectors,
                            #                                  return_label=True)
                            # data_builder_test = dataset_predict.get_Dataloader()
                            # X_test, y_test = data_builder_test.get_data_op_modelling(filename=dataset_name)
                        else:
                            X_test, y_test = data_builder_test.get_all_dataset_data_sr(query='last', label_files=None)
                        
                        logger.info(f'Start the creation of operator in itteration {j}')
                        operator_ = create_ML_model(X=X_train, y=y_train, name='perceptron_regression')
                    else:
                        dataset_train = Pipeline_Dataset(dataset_filenames, norm=True, create_operator=True, ret_class=True)
                        X_train, y_train = data_builder_train.get_all_dataset_data()

                        operator_ = create_operator(name=operator_name, X=X_train, y=y_train)
                    
                    dataset_pred_vectors = [dataset_pred_vectors]
                    if op.contains(operator_name.lower(), 'regression'):
                        if use_vectors:
                            r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, y_pred = predict_linear_regression_operator(operator=operator_, X=dataset_pred_vectors, y=y_test, ret_preds=True)
                        else:
                            r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, y_pred = predict_linear_regression_operator(
                                operator=operator_, X=X_test, y=y_test, ret_preds=True)
                        logger.info(
                            f'Operator name: {operator_name}, for dataset {dataset_name} r2 score is: {r2}, NRMSE: {nrmse_loss}, RMSE: {rmse_loss}, MAE : {mae_loss}, MAD : {mad_loss}, MaPE : {MaPE_loss}')
                        exec_time = time.time() - start_time
                        logger.info(f'Execution time : {exec_time}')
                        
                        pred_y.append(y_pred)
                        label_y.append(y_test)
                        if not return_res:
                            self.save_csv_file(header=['Operator Name', 'Data Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time'],
                                                record=[operator_name, dataset_name, '-', r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, exec_time],
                                                filename='results_out_new_vec.csv')
                        metrics_results_average['Exec time'] += exec_time

                    elif op.contains(operator_name.lower(), 'arima') or op.contains(operator_name.lower(), 'holt_winter'):
                        if use_vectors:
                            r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, y_pred = predict_linear_regression_operator(operator=operator_, X=dataset_pred_vectors, y=y_test, ret_preds=True)
                            acc = 0
                        else:
                            r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, y_pred = predict_time_series_model(
                                    operator=operator_, X=X_test, y=y_test, operator_name=operator_name, ret_preds=True)
                        logger.info('Use operator F')
                        # self.save_model(filename=operator_name, operator=operator_)
                        logger.info(
                                    f'Operator name: {operator_name}, for dataset {dataset_name} r2 score is: {r2}, NRMSE: {nrmse_loss}, RMSE: {rmse_loss}, MAE : {mae_loss}, MAD : {mad_loss}, MaPE : {MaPE_loss}')
                        exec_time = time.time() - start_time
                        logger.info(f'Execution time : {exec_time}')
                        if not return_res:
                            self.save_csv_file(header=['Operator Name', 'Data Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time'],
                                                record=[operator_name, dataset_name, '-', r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, exec_time],
                                                filename='results_out_new_vec.csv')
                        
                        pred_y.append(y_pred)
                        label_y.append(y_test)
                        # metrics_results_average['r^2'] += r2
                        # metrics_results_average['NRMSE'] += nrmse_loss
                        # metrics_results_average['RMSE'] += rmse_loss
                        # metrics_results_average['MAE'] += mae_loss
                        # metrics_results_average['MAD'] += mad_loss
                        # metrics_results_average['MaPE'] += MaPE_loss
                        metrics_results_average['Exec time'] += exec_time
                    else:
                        if use_vectors:
                            r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, y_pred = predict_linear_regression_operator(operator=operator_, X=dataset_pred_vectors, y=y_test, ret_preds=True)
                            acc = 0
                        else:
                            acc, r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, y_pred = predict_operator(operator=operator_, X=X_test, y=y_test, ret_preds=True)
                        logger.info(
                            f'Operator name: {operator_name}, for dataset {dataset_name} accuracy is: {acc}, r^2: {r2}, NRMSE: {nrmse_loss}, RMSE: {rmse_loss}, MAE : {mae_loss}, MAD : {mad_loss}, MaPE : {MaPE_loss}')
                        exec_time = time.time() - start_time
                        logger.info(f'Execution time : {exec_time}')
                        if not return_res:
                            self.save_csv_file(header=['Operator Name', 'Data Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time'],
                                                record=[operator_name, dataset_name, acc, '-', nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, exec_time],
                                                filename='results_out_new_vec.csv')
                        
                        pred_y.append(y_pred)
                        label_y.append(y_test)
                            
                        # metrics_results_average['NRMSE'] += nrmse_loss
                        # metrics_results_average['RMSE'] += rmse_loss
                        # metrics_results_average['MAE'] += mae_loss
                        # metrics_results_average['MAD'] += mad_loss
                        # metrics_results_average['MaPE'] += MaPE_loss
                        metrics_results_average['Exec time'] += exec_time
                        # metrics_results_average['Acc'] += acc
                    # self.save_model(filename=operator_name, operator=operator_)
        
        # print(pred_y)
        # print(label_y)

        label_y = [normalize_label(y) for y in label_y]

        pred_y = np.concatenate(pred_y, axis=0)
        label_y = np.concatenate(label_y, axis=0)
            
        print(pred_y.shape)
        print(label_y.shape)
        r_2_score, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss = get_metrics(y_pred=pred_y,
                                                                                      y_targets=label_y)
        
        metrics_results_average['NRMSE'] = nrmse_loss
        metrics_results_average['RMSE'] = rmse_loss
        metrics_results_average['MAE'] = mae_loss
        metrics_results_average['MAD'] = mad_loss
        metrics_results_average['MaPE'] = MaPE_loss
        metrics_results_average['Exec time'] /= count
        metrics_results_average['Acc'] = None 
        metrics_results_average['r^2'] = r_2_score

        if return_res:
            return metrics_results_average
        else:
            self.save_csv_file(header=['Operator Name', 'Data Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time'],
                                                record=[operator_name, 'Average All', metrics_results_average['Acc'], '-', metrics_results_average['NRMSE'], metrics_results_average['RMSE'], metrics_results_average['MAE'], metrics_results_average['MAD'], metrics_results_average['MaPE'], metrics_results_average['r^2']],
                                                filename='results_out_new_vec.csv')
            print('in')