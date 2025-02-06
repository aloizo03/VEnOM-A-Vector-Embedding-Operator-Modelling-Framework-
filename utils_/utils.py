import torch
import torch.nn as nn
import os
from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt

from permetrics.regression import RegressionMetric

from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.linear_model as linear
import sklearn.neural_network as neural_network
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import accuracy_score, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn import svm
import scipy

def calculate_accuracy(predictions, gt_labels):
    """
    Make predictions and calculate the accuracy
    :param predictions: torch.Tensor, a tensor with the model output predictions
    :param gt_labels:  torch.Tensor, a tensor with the ground ruth label for each prediction
    :return: float: the accuracy of the model prediction for this predictions
    """
    soft_max = torch.softmax(predictions, dim=-1)
    _, predicts = torch.max(soft_max, dim=-1, keepdim=False)
    corrects = (predicts == gt_labels).sum()
    corrects = corrects.cpu().detach().numpy()

    return corrects / gt_labels.size(0)


def check_path(path):
    if type(path) is list:
        for p in path:
             if not os.path.exists(p):
                AssertionError(f'Filepath {p} is not exist')
        return
    if not os.path.exists(path):
        AssertionError(f'Filepath {path} is not exist')


def get_parent_dir(filepath):
    head, tail = os.path.split(filepath)
    print(head, tail)

def calculate_cosine_distance(a, b):
    cosine_distance = float(spatial.distance.cosine(a, b))
    return cosine_distance
    
def calculate_cosine_similarity(a, b):
    cosine_similarity = 1 - calculate_cosine_distance(a, b)
    return cosine_similarity

def create_ML_model(name, X, y, n_layers=200):
    if name.lower() == 'perceptron':
        operator = MultiOutputClassifier(linear.SGDClassifier(max_iter=1000, tol=1e-3, loss='perceptron'))
        operator.fit(X=X, Y=y)
    elif name.lower() == 'perceptron_regression':
        operator = neural_network.MLPRegressor(hidden_layer_sizes=n_layers)

        operator.fit(X=X, y=y)
    return operator

def create_operator(name, X, y):
    if name.lower() == 'svm_sgd':
        operator = linear.SGDClassifier(max_iter=1000, tol=1e-3, loss='hinge')
    elif name.lower() == 'svm':
        operator = svm.SVC()
    elif name.lower() == 'svm_mcc':
        operator = svm.SVC(decision_function_shape='ovo')
    elif name.lower() == 'logistic_regression_sgd':
        operator = linear.SGDClassifier(max_iter=1000, tol=1e-3, loss='log_loss')
    elif name.lower() == 'logistic_regression':
        operator = linear.LogisticRegression()
    elif name.lower() == 'linear_regression':
        operator = linear.LinearRegression()
    elif name.lower() == 'perceptron':
        operator = linear.SGDClassifier(max_iter=1000, tol=1e-3, loss='perceptron')
    elif name.lower() == 'mlpregression':
        operator = neural_network.MLPRegressor()
    elif name.lower() == 'mlp_sgd':
        operator = neural_network.MLPClassifier(solver='sgd', alpha=1e-5, random_state=1)
    elif name.lower() == 'mlp_adam':
        operator = neural_network.MLPClassifier(solver='adam', alpha=1e-5, random_state=1)
    elif name.lower() == 'dbscan':
        operator = DBSCAN(eps=0.5, min_samples=5)  
    elif name.lower() == 'local_outlier_factor':
        operator = LocalOutlierFactor(n_neighbors=2)
    elif name.lower() == 'eigenvalue':
        operator = None
    elif name.lower() == 'arima':
        operator = ARIMA(y, order=(1,1,0))
        operator = operator.fit()
        return operator
    elif name.lower() == 'holt_winter':
        operator = ExponentialSmoothing(y, trend='add')
        operator = operator.fit()
        return operator
    
    operator.fit(X=X, y=y)

    
    return operator

def fit_operator(operator, operator_name, X, y):
    if operator_name.lower() == 'svm_sgd' or operator_name.lower() == 'logistic_regression_sgd' or operator_name.lower() == 'perceptron' or operator_name.lower() == 'mlpregression' or operator_name.lower() == 'mlp_sgd' or operator_name.lower() == 'mlp_adam' :
        operator = operator.partial_fit(X=X, y=y)
    elif operator_name.lower() == 'svm' or  operator_name.lower() == 'svm_mcc' or operator_name.lower() == 'logistic_regression':
        operator = operator.fit(X=X, y=y)
    elif operator_name.lower() == 'linear_regression':
        operator = operator.fit(X=X, y=y, coef_array=operator.coef_)
    elif operator_name.lower() == 'arima' or operator_name.lower() == 'holt_winter':
        operator = operator.fit(X)

    return operator


def predict_operator(operator, X, y):
    y_pred = operator.predict(X)

    if len(y_pred) < 2:
        y_pred = np.reshape(y_pred,(-1,y_pred.shape[0]))
        nrmse_loss = 1
    else:
        nrmse_loss = RegressionMetric(y, y_pred).normalized_root_mean_square_error()
    
    acc = accuracy_score(y_true=y, y_pred=y_pred)
    r_2_score = r2_score(y, y_pred)
    
    mae_loss = mean_absolute_error(y, y_pred)
    mad_loss = median_absolute_error(y, y_pred)
    rmse_loss = root_mean_squared_error(y, y_pred)
    MaPE_loss = mean_absolute_percentage_error(y, y_pred)
    return acc, r_2_score, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss

def predict_linear_regression_operator(operator, X, y, ret_preds=False):
    y_pred = operator.predict(X)
    
    if len(y_pred) < 2:
        y_pred = np.reshape(y_pred,(-1,y_pred.shape[0]))
        nrmse_loss = 1
    else:
        nrmse_loss = RegressionMetric(y, y_pred).normalized_root_mean_square_error()
        nrmse_loss = np.average(nrmse_loss)

    r2 = operator.score(X, y)
    mae_loss = mean_absolute_error(y, y_pred)
    mad_loss = median_absolute_error(y, y_pred)
    rmse_loss = root_mean_squared_error(y, y_pred)
    MaPE_loss = mean_absolute_percentage_error(y, y_pred)

    if ret_preds:
        return r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, y_pred
    else:
        return r2, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss


def predict_local_outlier(operator, X, y):
    y_pred = operator.fit_predict(X)
    acc = accuracy_score(y_true=y, y_pred=y_pred)
    r_2_score = r2_score(y, y_pred)
    nrmse_loss = RegressionMetric(y, y_pred).normalized_root_mean_square_error()
    mae_loss = mean_absolute_error(y, y_pred)
    mad_loss = median_absolute_error(y, y_pred)
    rmse_loss = root_mean_squared_error(y, y_pred)
    MaPE_loss = mean_absolute_percentage_error(y, y_pred)

    return acc, r_2_score, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss

def predict_time_series_model(operator, X, y, operator_name, ret_preds=False):
    if operator_name.lower() == 'arima':
        y_pred = operator.get_forecast(steps=X.shape[0])
        y_pred = np.asarray(y_pred.predicted_mean)
    elif operator_name.lower() == 'holt_winter':
        y_pred = operator.forecast(steps=X.shape[0])

    if len(y_pred) < 2:
        y_pred = np.reshape(y_pred,(-1,y_pred.shape[0]))
        nrmse_loss = 1
    else:
        nrmse_loss = RegressionMetric(y, y_pred).normalized_root_mean_square_error()
    
    r_2_score = r2_score(y, y_pred)
    mae_loss = mean_absolute_error(y, y_pred)
    mad_loss = median_absolute_error(y, y_pred)
    rmse_loss = root_mean_squared_error(y, y_pred)
    MaPE_loss = mean_absolute_percentage_error(y, y_pred)
    if ret_preds:
        return r_2_score, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, y_pred
    else:
        return r_2_score, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss

def predict_dbscan(operator, X, y):
    y_pred = operator.fit_predict(X=X)

    if len(y_pred) < 2:
        y_pred = np.reshape(y_pred,(-1,y_pred.shape[0]))
        nrmse_loss = 1
    else:
        nrmse_loss = RegressionMetric(y, y_pred).normalized_root_mean_square_error()
    
    acc = accuracy_score(y_true=y, y_pred=y_pred)
    r_2_score = r2_score(y, y_pred)
    
    mae_loss = mean_absolute_error(y, y_pred)
    mad_loss = median_absolute_error(y, y_pred)
    rmse_loss = root_mean_squared_error(y, y_pred)
    MaPE_loss = mean_absolute_percentage_error(y, y_pred)
    return acc, r_2_score, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss

def standardize_data(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    std_dev[std_dev == 0] = 1  # Prevent division by zero
    standardized = (data - mean) / std_dev

    return standardized

def compute_eig_value(X, y=None):

    standardized_data = standardize_data(X)
    if standardized_data.ndim == 1:
        standardized_data = standardized_data.reshape(-1, 1)

    # Step 3: Compute the covariance matrix
    cov_matrix = np.cov(standardized_data, rowvar=False)

    if cov_matrix.ndim == 1:
        cov_matrix = cov_matrix.reshape(1, -1)

    # Step 4: Compute eigenvalues and eigenvectors
    if cov_matrix.ndim == 0:
        eigenvalues = 0
        eigenvectors = 0
    else:
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    eig = np.max(eigenvalues)

    return eig


def compute_rank(X):
        # dataset = X[i]
    n_rows = X.shape[0]
    s = np.arange(1, n_rows + 1)
    # Calculate ranks
    ranks = np.argsort(np.argsort(X, axis=0), axis=0) + 1  # Rank each column
    overall_rank = ranks.mean(axis=1)  # Average rank if multiple columns
    # Compute the square root of the sum of squared differences
    y_pred = np.sqrt(np.sum((overall_rank - s) ** 2))
    return y_pred

def compute_sum(X):
   return np.sum(X)

def compute_avg(X):
    return np.average(X)