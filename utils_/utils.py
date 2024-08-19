import torch
import torch.nn as nn
import os
from scipy import spatial

import matplotlib.pyplot as plt

import sklearn.linear_model as linear
from sklearn.metrics import accuracy_score


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


def create_operator(name, X, y):
    if name.lower() == 'svm_sgd':
        operator = linear.SGDClassifier(max_iter=1000, tol=1e-3, loss='hinge')
    elif name.lower() == 'logistic_regression_sgd':
        operator = linear.SGDClassifier(max_iter=1000, tol=1e-3, loss='log_loss')
    elif name.lower() == 'linear_regression':
        operator = linear.LinearRegression()
    elif name.lower() == 'percepton':
        operator = linear.SGDClassifier(max_iter=1000, tol=1e-3, loss='perceptron')

    operator.fit(X=X, y=y)
    return operator


def predict_operator(operator, X, y):
    y_pred = operator.predict(X)
    acc = accuracy_score(y_true=y, y_pred=y_pred)

    return acc


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)