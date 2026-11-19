from data.Dataset import Pipeline_Dataset, graph_Pipeline_Dataset, Image_Pipeline_Dataset
from utils_.utils import get_metrics, create_operator, create_graph_operator, predict_operator, predict_linear_regression_operator, predict_time_series_model, predict_dbscan, create_ML_model, normalize_label, set_global_seed
import pickle
import logging
import os
import time
import operator as op
import csv
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

logger = logging.getLogger(__name__)

CURRENT_MACHINE_ROOT = os.environ.get('VENOM_RESULTS_ROOT', 'results/')


def _sanitize_test(X_test, y_test, dataset_name="unknown"):
    """
    Sanitize y_test and X_test by removing rows where y_test is NaN, inf, or None.
    """
    if not isinstance(y_test, np.ndarray):
        y_test = [float('nan') if (v is None or (isinstance(v, float) and np.isnan(v)))
                  else v for v in y_test]

    
    if np.ndim(y_test) == 0:
        y_test = np.array([y_test])

    y_arr = np.asarray(y_test, dtype=np.float64).flatten()
    x_arr = np.asarray(X_test, dtype=object) if not isinstance(X_test, np.ndarray) \
            else X_test

    try:
        x_arr = x_arr.astype(np.float64)
    except (ValueError, TypeError):
        pass

    # print(f"Sanitizing test data for dataset {dataset_name}: "
    #       f"X shape {np.shape(X_test)}, y shape {y_arr.shape}")

    mask = np.isfinite(y_arr)
    if mask.all():
        return x_arr, y_arr

    n_bad = int((~mask).sum())
    logger.warning(
        f'Removing {n_bad} NaN/inf/None test label(s) for dataset {dataset_name}.'
    )
    y_arr = y_arr[mask]

    if x_arr.ndim >= 2 and x_arr.shape[0] == len(mask):
        x_arr = x_arr[mask]
    elif x_arr.ndim == 1 and x_arr.shape[0] == len(mask):
        x_arr = x_arr[mask]
    return x_arr, y_arr


def _sanitize_train(X_train, y_train, dataset_name="unknown"):
    """
    Sanitize y_train and X_train by removing rows where y_train is NaN, inf, or None.
    """
    # ── coerce None → nan BEFORE the numpy cast ──────────────────────────────
    if not isinstance(y_train, np.ndarray):
        y_train = [float('nan') if (v is None or (isinstance(v, float) and np.isnan(v)))
                   else v for v in y_train]

    if np.ndim(y_train) == 0:
        y_train = np.array([y_train])

    y_arr = np.asarray(y_train, dtype=np.float64).flatten()
    x_arr = np.asarray(X_train, dtype=np.float64)

    mask = np.isfinite(y_arr)
    if mask.all():
        return x_arr, y_arr

    n_bad = int((~mask).sum())
    logger.warning(
        f'Removing {n_bad} NaN/inf/None training label(s) for dataset {dataset_name}.'
    )
    y_arr = y_arr[mask]
    if x_arr.ndim >= 2 and x_arr.shape[0] == len(mask):
        x_arr = x_arr[mask]
    elif x_arr.ndim == 1 and x_arr.shape[0] == len(mask):
        x_arr = x_arr[mask]
    return x_arr, y_arr

class Create_Operator:

    def __init__(self, out_path, operator_dir, out_file):
        self.out_path = out_path
        self.out_file = out_file
        self.operator_dir = operator_dir
        self.operator = self.read_operator()
        self.current_seed = None  # seed of the repetition being written (adds a 'Seed' column)
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
        if self.current_seed is not None:
            header = list(header) + ['Seed']
            record = list(record) + [self.current_seed]
        outfile = os.path.join(self.out_path, filename)
        if os.path.isfile(outfile):
            csv_file = open(outfile, 'a')
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow([r for r in record])
        else:
            with open(outfile, 'w') as csvfile:
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

    def make_path_agnostic(self, old_path, new_base_dir, anchor_folder="Data/"):
        if not isinstance(old_path, str):
            return old_path
        if anchor_folder in old_path:
            relative_path = old_path.split(anchor_folder)[-1]
            new_path = os.path.join(new_base_dir, anchor_folder, relative_path)
            return os.path.normpath(new_path)
        return old_path

    @staticmethod
    def aggregate_metrics_log(metrics_log):
        """
        metrics_log : list of tuples (r2, nrmse, rmse, mae, mad, mape).
        Returns dict with '<metric>_mean' and '<metric>_std' keys.
        """
        names = ['r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE']
        if len(metrics_log) == 0:
            return {f'{n}_mean': None for n in names} | {f'{n}_std': None for n in names}
        arr = np.array(metrics_log, dtype=float)
        stats = {}
        for i, name in enumerate(names):
            stats[f'{name}_mean'] = np.mean(arr[:, i])
            stats[f'{name}_std'] = np.std(arr[:, i], ddof=1) if arr.shape[0] > 1 else 0.0
        return stats

    @staticmethod
    def compute_wilcoxon_rmse_mae(metrics_log):
        """
        Wilcoxon signed-rank test between paired RMSE and MAE values -- one
        pair per (repetition, query dataset) entry in metrics_log (same
        ordering/columns as aggregate_metrics_log). Returns NaN for both
        stat and p if there are fewer than 2 pairs, or if every RMSE-MAE
        difference is exactly zero -- scipy.stats.wilcoxon raises a
        ValueError in that degenerate case rather than returning a result.
        """
        if len(metrics_log) < 2:
            return {'wilcoxon_stat': float('nan'), 'wilcoxon_p': float('nan')}
        arr = np.array(metrics_log, dtype=float)
        rmse_vals = arr[:, 2]
        mae_vals = arr[:, 3]
        try:
            stat, p = wilcoxon(rmse_vals, mae_vals)
        except ValueError:
            # e.g. "zero_method 'wilcox' and 'x - y == 0' for all elements"
            stat, p = float('nan'), float('nan')
        return {'wilcoxon_stat': float(stat), 'wilcoxon_p': float(p)}

    def create_operator(self, operator_name, most_relevant_data_path, target_labels=None,
                        repetitions=5, use_vectors=True, load_rel_data=True,
                        return_res=False, labels_dict=None, data_type=1, seeds=None):
        """seeds: optional list of ints. When given, one repetition is run per seed
        (repetitions = len(seeds)); before each repetition numpy/random/torch are
        seeded and the seed is passed as random_state to the surrogate, and every
        per-row CSV record gets a 'Seed' column so per-run results can be paired."""
        if seeds is not None:
            seeds = [int(s) for s in seeds]
            repetitions = len(seeds)

        if data_type == 2:
            anchor_path = 'Data/'
        elif data_type == 1:
            anchor_path = 'data/'
        elif data_type == 3:
            anchor_path = 'Image_Data/'

        if load_rel_data:
            selected_dic = self.load_dict(path_=most_relevant_data_path)
        else:
            selected_dic = most_relevant_data_path

        metrics_results_average = {
            'Acc': 0, 'r^2': 0, 'NRMSE': 0, 'RMSE': 0, 'MAE': 0, 'MAD': 0, 'MaPE': 0, 'Exec time': 0
        }
        metrics_log = []

        if labels_dict is not None and isinstance(labels_dict, str):
            labels_dict = pd.read_csv(labels_dict)
            if 'Dataset Name' in labels_dict.columns:
                col_name = 'Dataset Name'
            elif 'file_path' in labels_dict.columns:
                col_name = 'file_path'
            else:
                raise KeyError("Neither 'Dataset Name' nor 'filepath' found in the labels CSV.")
            labels_dict[col_name] = labels_dict[col_name].apply(
                lambda path: self.make_path_agnostic(path, CURRENT_MACHINE_ROOT, anchor_folder=anchor_path)
            )

        if target_labels is not None and isinstance(target_labels, str):
            labels_dict_target = pd.read_csv(target_labels)
            target_labels_path = target_labels
            if 'Dataset Name' in labels_dict_target.columns:
                col_name_target = 'Dataset Name'
            elif 'file_path' in labels_dict_target.columns:
                col_name_target = 'file_path'
            else:
                raise KeyError("Neither 'Dataset Name' nor 'file_path' found in the target labels CSV.")
            labels_dict_target[col_name_target] = labels_dict_target[col_name_target].apply(
                lambda path: self.make_path_agnostic(path, CURRENT_MACHINE_ROOT, anchor_folder=anchor_path)
            )

        count = 0
        most_relevant_data = selected_dic['selected_dataset']
        pred_vectors = selected_dic['Pred_Dataset']
        pred_y = []
        label_y = []

        for j in range(repetitions):
            self.current_seed = seeds[j] if seeds is not None else None
            if self.current_seed is not None:
                set_global_seed(self.current_seed)
            for dataset_name, dataset_list in most_relevant_data.items():
                count += 1

                # ---- load test data ----------------------------------------
                if data_type == 2:
                    dataset_pred_vectors = pred_vectors[dataset_name]
                    dataset_predict = graph_Pipeline_Dataset(data_path=dataset_name,
                                                             labels_dict=labels_dict_target,
                                                             Vectors=dataset_pred_vectors,
                                                             return_label=True)
                    data_builder_test = dataset_predict.get_Dataloader()
                    X_test, y_test = data_builder_test.get_all_data()


                elif data_type == 3:
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

                # sanitize test labels (shape-aware -- safe for both 2D
                # sample matrices and 1D embedding vectors)
                X_test, y_test = _sanitize_test(X_test, y_test, dataset_name)
                if len(y_test) == 0:
                    logger.warning(
                        f'Skipping dataset {dataset_name}: no valid test samples '
                        f'remaining after NaN/inf removal.'
                    )
                    count -= 1
                    continue
                # ---- pretrained operator branch ----------------------------
                if self.operator is not None:
                    start_time = time.time()
                    if op.contains(operator_name.lower(), 'regression'):
                        r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss = predict_linear_regression_operator(
                            operator=self.operator, X=X_test, y=y_test)
                        logger.info('Use operator F')
                        logger.info(
                            f'Operator name: {operator_name}, for dataset {dataset_name} '
                            f'r2={r2}, NRMSE={nrmse_loss}, RMSE={rmse_loss}, '
                            f'MAE={mae_loss}, MAD={mad_loss}, MaPE={MaPE_loss}')
                        exec_time = time.time() - start_time
                        logger.info(f'Execution time: {exec_time}')
                        metrics_log.append((r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss))
                        if not return_res:
                            self.save_csv_file(
                                header=['Operator Name', 'Data Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time'],
                                record=[operator_name, dataset_name, '-', r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, exec_time],
                                filename=self.out_file)
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
                            f'Operator name: {operator_name}, for dataset {dataset_name} '
                            f'r2={r2}, NRMSE={nrmse_loss}, RMSE={rmse_loss}, '
                            f'MAE={mae_loss}, MAD={mad_loss}, MaPE={MaPE_loss}')
                        exec_time = time.time() - start_time
                        logger.info(f'Execution time: {exec_time}')
                        metrics_log.append((r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss))
                        if not return_res:
                            self.save_csv_file(
                                header=['Operator Name', 'Data Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time'],
                                record=[operator_name, dataset_name, '-', r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, exec_time],
                                filename=self.out_file)
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
                        logger.info(f'Execution time: {exec_time}')
                        metrics_log.append((r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss))
                        if not return_res:
                            self.save_csv_file(
                                header=['Operator Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time'],
                                record=[operator_name, acc, '-', nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, exec_time],
                                filename=self.out_file)
                        metrics_results_average['r^2'] += r2
                        metrics_results_average['NRMSE'] += nrmse_loss
                        metrics_results_average['RMSE'] += rmse_loss
                        metrics_results_average['MAE'] += mae_loss
                        metrics_results_average['MAD'] += mad_loss
                        metrics_results_average['MaPE'] += MaPE_loss
                        metrics_results_average['Exec time'] += exec_time

                    else:
                        acc, r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss = predict_operator(
                            operator=self.operator, X=X_test, y=y_test)
                        logger.info('Use operator F, above threshold')
                        logger.info(
                            f'Operator name: {operator_name}, for dataset {dataset_name} '
                            f'acc={acc}, r2={r2}, NRMSE={nrmse_loss}, RMSE={rmse_loss}, '
                            f'MAE={mae_loss}, MAD={mad_loss}, MaPE={MaPE_loss}')
                        exec_time = time.time() - start_time
                        logger.info(f'Execution time: {exec_time}')
                        metrics_log.append((r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss))
                        if not return_res:
                            self.save_csv_file(
                                header=['Operator Name', 'Data Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time'],
                                record=[operator_name, dataset_name, acc, '-', nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, exec_time],
                                filename=self.out_file)
                        metrics_results_average['r^2'] += r2
                        metrics_results_average['NRMSE'] += nrmse_loss
                        metrics_results_average['RMSE'] += rmse_loss
                        metrics_results_average['MAE'] += mae_loss
                        metrics_results_average['MAD'] += mad_loss
                        metrics_results_average['MaPE'] += MaPE_loss
                        metrics_results_average['Exec time'] += exec_time
                        metrics_results_average['Acc'] += acc

                # ---- surrogate (no pretrained operator) branch -------------
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
                            # sanitize test (shape-aware)
                            X_test, y_test = _sanitize_test(X_test, y_test, dataset_name)
                            if len(y_test) == 0:
                                logger.warning(
                                    f'Skipping dataset {dataset_name}: no valid test samples '
                                    f'remaining after NaN/inf removal.'
                                )
                                count -= 1
                                continue
                        elif data_type == 2:
                            dataset_train = graph_Pipeline_Dataset(data_path=dataset_filenames,
                                                                   labels_dict=labels_dict,
                                                                   Vectors=dataset_vectors,
                                                                   return_label=True)
                            data_builder_train = dataset_train.get_Dataloader()
                            X_train, y_train = data_builder_train.get_all_data()
                        elif data_type == 3:
                            dataset_pred_vectors_local = pred_vectors[dataset_name]
                            dataset_train = Image_Pipeline_Dataset(dataset_filenames,
                                                                   labels_dict=labels_dict,
                                                                   Vectors=dataset_vectors,
                                                                   create_op=True,
                                                                   return_label=True)
                            data_builder_train = dataset_train.get_Dataloader()
                            X_train, y_train = data_builder_train.get_all_data_operator_modelling(dataset_filenames)
                        else:
                            X_test, y_test = data_builder_test.get_all_dataset_data_sr(query='last', label_files=None)
                            X_test, y_test = _sanitize_test(X_test, y_test, dataset_name)
                            if len(y_test) == 0:
                                logger.warning(
                                    f'Skipping dataset {dataset_name}: no valid test samples '
                                    f'remaining after NaN/inf removal.'
                                )
                                count -= 1
                                continue
                        logger.info(f'Start the creation of operator in iteration {j}')

                        # sanitize train labels (always a 2D matrix, safe to filter)
                        X_train, y_train = _sanitize_train(X_train, y_train, dataset_name)
                        
                        if len(y_train) == 0:
                            logger.warning(
                                f'Skipping dataset {dataset_name}: no valid training samples '
                                f'remaining after NaN/inf removal. Decrementing count.'
                            )
                            count -= 1
                            continue
                        
                        if data_type == 1:
                            operator_ = create_ML_model(X=X_train, y=y_train, name=operator_name, seed=self.current_seed)
                        elif data_type == 2:
                            operator_ = create_ML_model(X=X_train, y=y_train, name='mlp_regression', seed=self.current_seed)
                        elif data_type == 3:
                            operator_ = create_ML_model(X=X_train, y=y_train, name='mlp_regression', seed=self.current_seed)

                    else:
                        dataset_train = Pipeline_Dataset(dataset_filenames, norm=True, create_operator=True, ret_class=True)
                        X_train, y_train = data_builder_train.get_all_dataset_data()
                        
                        if data_type == 1:
                            operator_ = create_ML_model(X=X_train, y=y_train, name=operator_name, seed=self.current_seed)
                        elif data_type == 2:
                            operator_ = create_ML_model(X=X_train, y=y_train, name='mlp_regression', seed=self.current_seed)
                        elif data_type == 3:
                            operator_ = create_ML_model(X=X_train, y=y_train, name='mlp_regression', seed=self.current_seed)

                    dataset_pred_vectors = [dataset_pred_vectors]

                    if op.contains(operator_name.lower(), 'regression'):
                        if use_vectors:
                            r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, y_pred = predict_linear_regression_operator(
                                operator=operator_, X=dataset_pred_vectors, y=y_test, ret_preds=True)
                        else:
                            r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, y_pred = predict_linear_regression_operator(
                                operator=operator_, X=X_test, y=y_test, ret_preds=True)
                        logger.info(
                            f'Operator name: {operator_name}, for dataset {dataset_name} '
                            f'r2={r2}, NRMSE={nrmse_loss}, RMSE={rmse_loss}, '
                            f'MAE={mae_loss}, MAD={mad_loss}, MaPE={MaPE_loss}')
                        exec_time = time.time() - start_time
                        logger.info(f'Execution time: {exec_time}')
                        metrics_log.append((r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss))
                        pred_y.append(y_pred)
                        label_y.append(y_test)
                        if not return_res:
                            self.save_csv_file(
                                header=['Operator Name', 'Data Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time'],
                                record=[operator_name, dataset_name, '-', r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, exec_time],
                                filename=self.out_file)
                        metrics_results_average['Exec time'] += exec_time

                    elif op.contains(operator_name.lower(), 'arima') or op.contains(operator_name.lower(), 'holt_winter'):
                        if use_vectors:
                            r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, y_pred = predict_linear_regression_operator(
                                operator=operator_, X=dataset_pred_vectors, y=y_test, ret_preds=True)
                            acc = 0
                        else:
                            r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, y_pred = predict_time_series_model(
                                operator=operator_, X=X_test, y=y_test, operator_name=operator_name, ret_preds=True)
                        logger.info('Use operator F')
                        logger.info(
                            f'Operator name: {operator_name}, for dataset {dataset_name} '
                            f'r2={r2}, NRMSE={nrmse_loss}, RMSE={rmse_loss}, '
                            f'MAE={mae_loss}, MAD={mad_loss}, MaPE={MaPE_loss}')
                        exec_time = time.time() - start_time
                        logger.info(f'Execution time: {exec_time}')
                        metrics_log.append((r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss))
                        if not return_res:
                            self.save_csv_file(
                                header=['Operator Name', 'Data Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time'],
                                record=[operator_name, dataset_name, '-', r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, exec_time],
                                filename=self.out_file)
                        pred_y.append(y_pred)
                        label_y.append(y_test)
                        metrics_results_average['Exec time'] += exec_time

                    else:
                        if use_vectors:
                            r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, y_pred = predict_linear_regression_operator(
                                operator=operator_, X=dataset_pred_vectors, y=y_test, ret_preds=True)
                            acc = 0
                        else:
                            acc, r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, y_pred = predict_operator(
                                operator=operator_, X=X_test, y=y_test, ret_preds=True)
                        logger.info(
                            f'Operator name: {operator_name}, for dataset {dataset_name} '
                            f'acc={acc}, r2={r2}, NRMSE={nrmse_loss}, RMSE={rmse_loss}, '
                            f'MAE={mae_loss}, MAD={mad_loss}, MaPE={MaPE_loss}')
                        exec_time = time.time() - start_time
                        logger.info(f'Execution time: {exec_time}')
                        metrics_log.append((r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss))
                        if not return_res:
                            self.save_csv_file(
                                header=['Operator Name', 'Data Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time'],
                                record=[operator_name, dataset_name, acc, '-', nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, exec_time],
                                filename=self.out_file)
                        pred_y.append(y_pred)
                        label_y.append(y_test)
                        metrics_results_average['Exec time'] += exec_time

        self.current_seed = None  # summary rows below carry no seed
        label_y = [normalize_label(y) for y in label_y]
        pred_y = np.concatenate(pred_y, axis=0)
        label_y = np.concatenate(label_y, axis=0)

        print(pred_y.shape)
        print(label_y.shape)
        r_2_score, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss = get_metrics(
            y_pred=pred_y, y_targets=label_y)

        metrics_results_average['NRMSE'] = nrmse_loss
        metrics_results_average['RMSE'] = rmse_loss
        metrics_results_average['MAE'] = mae_loss
        metrics_results_average['MAD'] = mad_loss
        metrics_results_average['MaPE'] = MaPE_loss
        metrics_results_average['Exec time'] /= count
        metrics_results_average['Acc'] = None
        metrics_results_average['r^2'] = r_2_score

        rep_stats = self.aggregate_metrics_log(metrics_log)
        metrics_results_average.update(rep_stats)

        # Wilcoxon signed-rank test, RMSE vs MAE, paired over every
        # (repetition, query dataset) entry collected across the repetitions
        # loop above.
        wilcoxon_stats = self.compute_wilcoxon_rmse_mae(metrics_log)
        metrics_results_average.update(wilcoxon_stats)

        if return_res:
            return metrics_results_average
        else:
            summary_filename = os.path.splitext(self.out_file)[0] + '_summary.csv'
            self.save_csv_file(
                header=['Operator Name', 'Data Name', 'Acc', 'r^2', 'NRMSE', 'RMSE', 'MAE', 'MAD', 'MaPE', 'Exec time',
                        'r^2_mean', 'r^2_std', 'NRMSE_mean', 'NRMSE_std', 'RMSE_mean', 'RMSE_std',
                        'MAE_mean', 'MAE_std', 'MAD_mean', 'MAD_std', 'MaPE_mean', 'MaPE_std',
                        'wilcoxon_stat', 'wilcoxon_p'],
                record=[operator_name, 'Average All',
                        metrics_results_average['Acc'], metrics_results_average['r^2'],
                        metrics_results_average['NRMSE'], metrics_results_average['RMSE'],
                        metrics_results_average['MAE'], metrics_results_average['MAD'],
                        metrics_results_average['MaPE'], metrics_results_average['Exec time'],
                        metrics_results_average['r^2_mean'], metrics_results_average['r^2_std'],
                        metrics_results_average['NRMSE_mean'], metrics_results_average['NRMSE_std'],
                        metrics_results_average['RMSE_mean'], metrics_results_average['RMSE_std'],
                        metrics_results_average['MAE_mean'], metrics_results_average['MAE_std'],
                        metrics_results_average['MAD_mean'], metrics_results_average['MAD_std'],
                        metrics_results_average['MaPE_mean'], metrics_results_average['MaPE_std'],
                        metrics_results_average['wilcoxon_stat'], metrics_results_average['wilcoxon_p']],
                filename=summary_filename)
            print('in')