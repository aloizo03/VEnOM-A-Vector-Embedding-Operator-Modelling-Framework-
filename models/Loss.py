import numpy as np
import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class Loss_Compute:

    def __init__(self, device, batch_size=64, normalize_loss=False, temperature=10,
                 base_temperature=10, criterion=None):
        self.device = device
        self.criterion = criterion
        self.norm_loss = normalize_loss
        self.batch_size = batch_size
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.criterion = criterion

    def __call__(self, feat_project, labels_y):
        labels_y = labels_y.contiguous().view(-1, 1)
        mask_ = torch.eq(labels_y, labels_y.T).float().to(labels_y.device)

        contrast_count = feat_project.shape[1]
        contrast_feat = torch.cat(torch.unbind(feat_project, dim=0), dim=0)

        anchor_feature = contrast_feat
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feat.T),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=0, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask_ = mask_.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask_, dtype=torch.int64),
            1,
            torch.arange(self.batch_size * anchor_count, dtype=torch.int64).view(-1, 1).to(feat_project.device),
            0, )
        mask_ = mask_ * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask_ * log_prob).sum(1) / mask_.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, self.batch_size).mean()
        if self.criterion is not None:
            loss = loss + cretirion(mask_, log_prob)
        return loss


class JointLoss(nn.Module):

    def __init__(self):
        super().__init__()
        pass
