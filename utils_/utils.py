import torch
import torch.nn as nn


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
