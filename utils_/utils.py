import torch
import torch.nn as nn
import os
from scipy import spatial
import numpy as np

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
import networkx as nx
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn import svm
import scipy

from server.server_utils.qdrant_controller import qdrant_controller

def get_file_modified_time(file):
    return os.stat(file).st_ctime

def get_vec_DB_collections():
    controler = qdrant_controller()
    return controler.get_all_colections()

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
        operator = neural_network.MLPRegressor(hidden_layer_sizes=n_layers, max_iter=1000, early_stopping=True)
        operator.fit(X=X, y=y)
    return operator

def freeman_generalization(centrality_dict):
    values = np.array(list(centrality_dict.values()))
    
    if values.size == 0:
        return 0.0
    
    max_c = np.max(values)
    n = len(values)
    if n <= 1:
        return 0
    return np.sum(max_c - values) / (n - 1)

# --- Spectral Radius ---
def spectral_radius(G):
    A = nx.to_numpy_array(G)
    eigenvalues = np.linalg.eigvals(A)
    return np.max(np.abs(eigenvalues))

def page_rank(G):
    G = nx.convert_node_labels_to_integers(G)
    max_pr =  max(nx.pagerank(G).values())
    return max_pr

def img_svm(x_train,y_train, X_test, sgd=False):
    # Defining the parameters grid for GridSearchCV
    param_grid={'C':[0.1,1,10,100],
                'gamma':[0.0001,0.001,0.1,1],
                'kernel':['rbf','poly']}

    # Creating a support vector classifier
    svc=svm.SVC(probability=True, decision_function_shape='ovo')

    # Creating a model using GridSearchCV with the parameters grid
    model=GridSearchCV(svc,param_grid)

    model.fit(x_train,y_train)

    return model, model.predict(X_test)

def create_image_operator(name, X, y, X_test):
    y_ret = []
    if name.lower() == 'img_svm' or name.lower() == 'image_svm':
        pass
    
def create_graph_operator(name, graph_list):
    y = []

    for graph in graph_list:
        # Load graph
        G = nx.read_edgelist(graph, nodetype=int)

        # Skip empty graph safely
        if G.number_of_nodes() == 0:
            y.append([0.0])
            continue

        # Normalize labels
        G = nx.convert_node_labels_to_integers(G)

        # Betweenness centrality
        if name.lower() == 'bc':
            c = nx.betweenness_centrality(G)
            y.append([freeman_generalization(c)])
            continue

        # Edge betweenness
        if name.lower() == 'ebc':
            c = nx.edge_betweenness_centrality(G)
            y.append([freeman_generalization(c)])
            continue

        # Closeness
        if name.lower() == 'cc':
            c = nx.closeness_centrality(G)
            y.append([freeman_generalization(c)])
            continue

        # Eigenvector centrality (needs special handling)
        if name.lower() == 'ec':
            try:
                c = nx.eigenvector_centrality_numpy(G)
                y.append([freeman_generalization(c)])
            except Exception:
                # Graph is disconnected or empty → return safe value
                y.append([0.0])
            continue

        # PageRank
        if name.lower() == 'pr':
            try:
                y.append([max(nx.pagerank(G).values())])
            except Exception:
                y.append([0.0])
            continue

        # Spectral radius
        if name.lower() == 'sr':
            try:
                y.append([spectral_radius(G)])
            except Exception:
                y.append([0.0])
            continue
    return y

def create_operator(name, X, y):
    print(name)
    if name.lower() == 'svm_sgd':
        operator = linear.SGDClassifier(max_iter=1000, tol=1e-3, loss='hinge')
    elif name.lower() == 'svm':
        operator = svm.SVC()
    elif name.lower() == 'svm_mcc':
        operator = svm.SVC(decision_function_shape='ovo', max_iter=1000, tol=1e-3)
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
    elif name.lower() == 'knn':
        operator = KNeighborsClassifier(n_neighbors=3)
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


def predict_operator(operator, X, y, ret_preds=False):
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
    if ret_preds:
        return acc, r_2_score, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss, y_pred
    else:
        return acc, r_2_score, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss
    
def predict_linear_regression_operator(operator, X, y, ret_preds=False):
    
    if not isinstance(y, (list, np.ndarray)):
        y = np.array([y])
        y[np.isnan(y)] = 0

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

def get_metrics(y_pred, y_targets):
    
    nrmse_loss = RegressionMetric(y_true=y_targets, y_pred=y_pred).normalized_root_mean_square_error()
    nrmse_loss = np.average(nrmse_loss)

    r_2_score = r2_score(y_true=y_targets, y_pred=y_pred)
    mae_loss = mean_absolute_error(y_true=y_targets, y_pred=y_pred)
    mad_loss = median_absolute_error(y_true=y_targets, y_pred=y_pred)
    rmse_loss = root_mean_squared_error(y_true=y_targets, y_pred=y_pred)
    MaPE_loss = mean_absolute_percentage_error(y_true=y_targets, y_pred=y_pred)
    
    return r_2_score, nrmse_loss, rmse_loss, mae_loss, mad_loss, MaPE_loss
    

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

    cov_matrix = np.cov(standardized_data, rowvar=False)

    if cov_matrix.ndim == 1:
        cov_matrix = cov_matrix.reshape(1, -1)

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

import os

def safe_join(base, rel):
    base_parts = os.path.normpath(base).split(os.sep)
    rel_parts = os.path.normpath(rel).lstrip(os.sep).split(os.sep)

    # Avoid duplication if last folder in base == first folder in rel
    if base_parts and rel_parts and base_parts[-1] == rel_parts[0]:
        rel_parts = rel_parts[1:]

    return os.path.join(base, *rel_parts)



def update_img_fullpath(df, filepath):
    head_path = os.path.dirname(filepath)
    head_path_abs = os.path.abspath(head_path)

    # Determine which column to use
    col = "file_path" if "file_path" in df.columns else "filename"

    for index, row in df.iterrows():
        original = str(row[col]).strip()

        # Normalize existing string
        normalized = os.path.normpath(original)

        # Absolute paths for reliable comparison
        normalized_abs = os.path.abspath(normalized)

        # Case 1: Already inside head_path → keep it as-is
        if normalized_abs.startswith(head_path_abs):
            df.at[index, col] = normalized_abs
            continue

        # Case 2: Filename already contains the dataset folder "f-MNIST"
        # Avoid:

        #   head_path / "f-MNIST/img1.png" → duplicate!
        last_folder = os.path.basename(head_path_abs)

        parts = normalized.split(os.sep)
        if parts[0] == last_folder:
            # Remove duplicate dataset folder
            normalized = os.path.join(*parts[1:])

        # Now safely join
        full_path = os.path.normpath(os.path.join(head_path_abs, normalized))

        df.at[index, col] = full_path

    return df

import numpy as np

def normalize_label(y):
    """
    Normalize label into a 1-D numpy array.
    Handles scalars, lists, nested lists of size N.
    """
    arr = np.array(y)

    # Case 0: scalar → [scalar]
    if arr.ndim == 0:
        return arr.reshape(1)

    # Case 1: shape (N, 1) → (N,)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.reshape(-1)

    # Case 2: already 1D → (N,)
    if arr.ndim == 1:
        return arr

    # Case 3: shape (1, N) → (N,)
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr.flatten()

    # Unexpected shape (rare)
    return arr.flatten()
