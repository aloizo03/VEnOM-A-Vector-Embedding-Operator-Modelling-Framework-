from data.Dataset import Pipeline_Dataset
from utils_.utils import create_operator, predict_operator, predict_linear_regression_operator, predict_time_series_model, predict_dbscan, create_ML_model
import pickle
import logging
import os
import time
import scipy
import operator as op
import csv

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
                # Gives the header name row into csv
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

    # def create_operator(self, operator_name, most_relevant_data_path, threshold):
    #     most_relevant_data = self.load_dict(path_=most_relevant_data_path)

    #     for dataset_name, dataset_list in most_relevant_data.items():
    #         dataset_predict = Pipeline_Dataset(dataset_name, norm=True, create_operator=True, ret_class=True)
    #         data_builder_test = dataset_predict.get_Dataloader()
    #         X_test, y_test = data_builder_test.get_all_dataset_data()
    #         if self.operator is not None:
    #             start_time = time.time()
    #             if op.contains(operator_name.lower(), 'regression'):
    #                 r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss = predict_linear_regression_operator(
    #                     operator=self.operator, X=X_test, y=y_test)
    #                 if rmse_loss < threshold:
    #                     logger.info('Use operator F, above threshold')
    #                     self.save_model(filename=operator_name, operator=self.operator)
    #                     logger.info(
    #                         f'Operator name: {operator_name}, for dataset {dataset_name} r2 score is: {r2}, NRMSE: {nrmse_loss}, RMSE: {rmse_loss}, MAE : {mae_loss}, MAD : {mad_loss}, MaPE : {MaPE_loss}')
    #                     logger.info(f'Execution time : {time.time() - start_time}')
    #                     return

    #             else:
    #                 acc, r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss = predict_operator(operator=self.operator,
    #                                                                                                  X=X_test, y=y_test)

    #                 logger.info(f'Accuracy: {acc}')
    #                 if acc > threshold:
    #                     logger.info('Use operator F, above threshold')
    #                     self.save_model(filename=operator_name, operator=self.operator)
    #                     logger.info(
    #                         f'Operator name: {operator_name}, for dataset {dataset_name} accuracy is: {acc}, r^2: {r2}, NRMSE: {nrmse_loss}, RMSE: {rmse_loss}, MAE : {mae_loss}, MAD : {mad_loss}, MaPE : {MaPE_loss}')
    #                     logger.info(f'Execution time : {time.time() - start_time}')
    #                     return

    #         start_time = time.time()
    #         dataset_train = Pipeline_Dataset(dataset_list, norm=True, create_operator=True, ret_class=True)
    #         data_builder_train = dataset_train.get_Dataloader()
    #         X_train, y_train = data_builder_train.get_all_dataset_data()
    #         operator_ = create_operator(name=operator_name,
    #                                     X=X_train,
    #                                     y=y_train)

    #         if op.contains(operator_name.lower(), 'regression'):
    #             r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss = predict_linear_regression_operator(
    #                 operator=operator_, X=X_test, y=y_test)
    #             logger.info(
    #                 f'Operator name: {operator_name}, for dataset {dataset_name} r2 score is: {r2}, NRMSE: {nrmse_loss}, RMSE: {rmse_loss}, MAE : {mae_loss}, MAD : {mad_loss}, MaPE : {MaPE_loss}')
    #             logger.info(f'Execution time : {time.time() - start_time}')
    #         else:
    #             acc, r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss = predict_operator(operator=operator_,
    #                                                                                              X=X_test, y=y_test)
    #             logger.info(
    #                 f'Operator name: {operator_name}, for dataset {dataset_name} accuracy is: {acc}, r^2: {r2}, NRMSE: {nrmse_loss}, RMSE: {rmse_loss}, MAE : {mae_loss}, MAD : {mad_loss}, MaPE : {MaPE_loss}')
    #             logger.info(f'Execution time : {time.time() - start_time}')
    #         self.save_model(filename=operator_name, operator=operator_)

    def create_operator(self, operator_name, most_relevant_data_path, threshold, repetitions=5, use_vectors=True):
        selected_dic = self.load_dict(path_=most_relevant_data_path)
        metrics_results_average = {
            'Acc': 0, 'r^2': 0, 'NRMSE': 0, 'RMSE':0 , 'MAE':0, 'MAD': 0, 'MaPE': 0, 'Exec time': 0
        }
        count = 0
        most_relevant_data = selected_dic['selected_dataset']
        pred_vectors = selected_dic['Pred_Dataset']
        for j in range(repetitions):
            for dataset_name, dataset_list in most_relevant_data.items():
                count += 1
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
                        # self.save_model(filename=operator_name, operator=self.operator)
                        logger.info(
                                    f'Operator name: {operator_name}, for dataset {dataset_name} r2 score is: {r2}, NRMSE: {nrmse_loss}, RMSE: {rmse_loss}, MAE : {mae_loss}, MAD : {mad_loss}, MaPE : {MaPE_loss}')
                        exec_time = time.time() - start_time
                        logger.info(f'Execution time : {exec_time}')
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
                        dataset_train = Pipeline_Dataset(dataset_filenames, norm=True, create_operator=False, ret_class=True, Vectors=dataset_vectors)
                        data_builder_train = dataset_train.get_Dataloader()
                        X_train, y_train = data_builder_train.get_all_dataset_data_sr(query='last', label_files=None)
                        dataset_predict = Pipeline_Dataset(dataset_name, norm=True, create_operator=False, ret_class=True, Vectors=dataset_pred_vectors)
                        data_builder_test = dataset_predict.get_Dataloader()
                        X_test, y_test = data_builder_test.get_all_dataset_data()
                        X_test, y_test = data_builder_test.get_all_dataset_data_sr(query='last', label_files=None)
                        operator_ = create_ML_model(X=X_train, y=y_train, name='perceptron_regression')
                    else:
                        dataset_train = Pipeline_Dataset(dataset_filenames, norm=True, create_operator=True, ret_class=True)
                        X_train, y_train = data_builder_train.get_all_dataset_data()

                        operator_ = create_operator(name=operator_name, X=X_train, y=y_train)
                    dataset_pred_vectors = [dataset_pred_vectors]
                    if op.contains(operator_name.lower(), 'regression'):
                        if use_vectors:
                            r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss = predict_linear_regression_operator(operator=operator_, X=dataset_pred_vectors, y=y_test)
                        else:
                            r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss = predict_linear_regression_operator(
                                operator=operator_, X=X_test, y=y_test)
                        logger.info(
                            f'Operator name: {operator_name}, for dataset {dataset_name} r2 score is: {r2}, NRMSE: {nrmse_loss}, RMSE: {rmse_loss}, MAE : {mae_loss}, MAD : {mad_loss}, MaPE : {MaPE_loss}')
                        exec_time = time.time() - start_time
                        logger.info(f'Execution time : {exec_time}')
                        self.save_csv_file(header=['Operator Name', 'Data Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time'],
                                            record=[operator_name, dataset_name, '-', r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, exec_time],
                                            filename='results_out_new_vec.csv')
                        metrics_results_average['r^2'] += r2
                        metrics_results_average['NRMSE'] += nrmse_loss
                        metrics_results_average['RMSE'] += rmse_loss
                        metrics_results_average['MAE'] += mae_loss
                        metrics_results_average['MAD'] += mad_loss
                        metrics_results_average['MaPE'] += MaPE_loss
                        metrics_results_average['Exec time'] += exec_time

                    elif op.contains(operator_name.lower(), 'arima') or op.contains(operator_name.lower(), 'holt_winter'):
                        r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss = predict_time_series_model(
                                operator=operator_, X=X_test, y=y_test, operator_name=operator_name)
                        logger.info('Use operator F')
                        # self.save_model(filename=operator_name, operator=operator_)
                        logger.info(
                                    f'Operator name: {operator_name}, for dataset {dataset_name} r2 score is: {r2}, NRMSE: {nrmse_loss}, RMSE: {rmse_loss}, MAE : {mae_loss}, MAD : {mad_loss}, MaPE : {MaPE_loss}')
                        exec_time = time.time() - start_time
                        logger.info(f'Execution time : {exec_time}')
                        self.save_csv_file(header=['Operator Name', 'Data Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time'],
                                            record=[operator_name, dataset_name, '-', r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, exec_time],
                                            filename='results_out_new_vec.csv')
                        metrics_results_average['r^2'] += r2
                        metrics_results_average['NRMSE'] += nrmse_loss
                        metrics_results_average['RMSE'] += rmse_loss
                        metrics_results_average['MAE'] += mae_loss
                        metrics_results_average['MAD'] += mad_loss
                        metrics_results_average['MaPE'] += MaPE_loss
                        metrics_results_average['Exec time'] += exec_time
                    else:
                        if use_vectors:
                            r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss = predict_linear_regression_operator(operator=operator_, X=dataset_pred_vectors, y=y_test)
                            acc = 0
                        else:
                            acc, r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss = predict_operator(operator=operator_, X=X_test, y=y_test)
                        logger.info(
                            f'Operator name: {operator_name}, for dataset {dataset_name} accuracy is: {acc}, r^2: {r2}, NRMSE: {nrmse_loss}, RMSE: {rmse_loss}, MAE : {mae_loss}, MAD : {mad_loss}, MaPE : {MaPE_loss}')
                        exec_time = time.time() - start_time
                        logger.info(f'Execution time : {exec_time}')
                        self.save_csv_file(header=['Operator Name', 'Data Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time'],
                                            record=[operator_name, dataset_name, acc, '-', nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, exec_time],
                                            filename='results_out_new_vec.csv')
                        metrics_results_average['NRMSE'] += nrmse_loss
                        metrics_results_average['RMSE'] += rmse_loss
                        metrics_results_average['MAE'] += mae_loss
                        metrics_results_average['MAD'] += mad_loss
                        metrics_results_average['MaPE'] += MaPE_loss
                        metrics_results_average['Exec time'] += exec_time
                        metrics_results_average['Acc'] += acc
                    # self.save_model(filename=operator_name, operator=operator_)
        
        metrics_results_average['NRMSE'] /= count
        metrics_results_average['RMSE'] /= count
        metrics_results_average['MAE'] /= count
        metrics_results_average['MAD'] /= count
        metrics_results_average['MaPE'] /= count
        metrics_results_average['Exec time'] /= count
        metrics_results_average['Acc'] /= count
        metrics_results_average['r^2'] /= count

        self.save_csv_file(header=['Operator Name', 'Data Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time'],
                                            record=[operator_name, 'Average All', metrics_results_average['Acc'], '-', metrics_results_average['NRMSE'], metrics_results_average['RMSE'], metrics_results_average['MAE'], metrics_results_average['MAD'], metrics_results_average['MaPE'], metrics_results_average['r^2']],
                                            filename='results_out_new_vec.csv')