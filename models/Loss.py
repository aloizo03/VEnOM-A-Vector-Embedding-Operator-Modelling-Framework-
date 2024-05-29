import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class Loss_Compute:
    #TODO: fix it error in the
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

    # From SubTrack https://github.com/AstraZeneca/SubTab/blob/main/utils/loss_functions.py#L13
    def __init__(self, batch_size, temperature, device, cosine_similarity=True, reconstruction=False,
                 contrastive_loss=False, distance_loss=False):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.reconstruction = reconstruction
        self.contrastive_loss = contrastive_loss
        self.distance_loss = distance_loss
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_neg_samples = self._get_mask_for_neg_samples().type(torch.bool)
        self.similarity_fun = self._cosine_simililarity if cosine_similarity else self._dot_simililarity
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    @staticmethod
    def _dot_simililarity(x, y):
        # Reshape x: (2N, C) -> (2N, 1, C)
        x = x.unsqueeze(1)
        # Reshape y: (2N, C) -> (1, C, 2N)
        y = y.T.unsqueeze(0)
        # Similarity shape: (2N, 2N)
        similarity = th.tensordot(x, y, dims=2)
        return similarity

    def _cosine_simililarity(self, x, y):
        similarity = th.nn.CosineSimilarity(dim=-1)
        # Reshape x: (2N, C) -> (2N, 1, C)
        x = x.unsqueeze(1)
        # Reshape y: (2N, C) -> (1, C, 2N)
        y = y.unsqueeze(0)
        # Similarity shape: (2N, 2N)
        return similarity(x, y)

    def _get_mask_for_neg_samples(self):
        # Diagonal 2Nx2N identity matrix, which consists of four (NxN) quadrants
        diagonal = np.eye(2 * self.batch_size)
        # Diagonal 2Nx2N matrix with 1st quadrant being identity matrix
        q1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        # Diagonal 2Nx2N matrix with 3rd quadrant being identity matrix
        q3 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        # Generate mask with diagonals of all four quadrants being 1.
        mask = th.from_numpy((diagonal + q1 + q3))
        # Reverse the mask: 1s become 0, 0s become 1. This mask will be used to select negative samples
        mask = (1 - mask).type(th.bool)
        # Transfer the mask to the device and return
        return mask.to(self.device)

    def XNegloss(self, representation):
        # Compute similarity matrix
        similarity = self.similarity_fn(representation, representation)
        # Get similarity scores for the positive samples from the diagonal of the first quadrant in 2Nx2N matrix
        l_pos = th.diag(similarity, self.batch_size)
        # Get similarity scores for the positive samples from the diagonal of the third quadrant in 2Nx2N matrix
        r_pos = th.diag(similarity, -self.batch_size)
        # Concatenate all positive samples as a 2nx1 column vector
        positives = th.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        # Get similarity scores for the negative samples (samples outside diagonals in 4 quadrants in 2Nx2N matrix)
        negatives = similarity[self.mask_for_neg_samples].view(2 * self.batch_size, -1)
        # Concatenate positive samples as the first column to negative samples array
        logits = th.cat((positives, negatives), dim=1)
        # Normalize logits via temperature
        logits /= self.temperature
        # Labels are all zeros since all positive samples are the 0th column in logits array.
        # So we will select positive samples as numerator in NTXentLoss
        labels = th.zeros(2 * self.batch_size).to(self.device).long()
        # Compute total loss
        loss = self.criterion(logits, labels)
        # Loss per sample
        closs = loss / (2 * self.batch_size)
        # Return contrastive loss
        return closs

    def forward(self, representation, xrecon, xorig):
        """

        Args:
            representation (torch.FloatTensor):
            xrecon (torch.FloatTensor):
            xorig (torch.FloatTensor):

        """

        # recontruction loss
        recon_loss = torch.nn.MSELoss(xrecon, xorig) if self.reconstruction else torch.nn.BCELoss(xrecon, xorig)

        # Initialize contrastive and distance losses with recon_loss as placeholder
        closs, zrecon_loss = recon_loss, recon_loss

        # Start with default loss i.e. reconstruction loss
        loss = recon_loss

        if self.contrastive_loss:
            closs = self.XNegloss(representation)
            loss = loss + closs

        if self.distance_loss:
            # recontruction loss for z
            zi, zj = th.split(representation, self.batch_size)
            zrecon_loss = getMSEloss(zi, zj)
            loss = loss + zrecon_loss

        # Return
        return loss, closs, recon_loss, zrecon_loss


class XNegLoss(nn.Module):

    def __init__(self, batch_size, device, cosine_similarity=True):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_neg_samples = self._get_mask_for_neg_samples().type(torch.bool)
        self.similarity_fun = self._cosine_simililarity if cosine_similarity else self._dot_simililarity
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    @staticmethod
    def _dot_simililarity(x, y):
        # Reshape x: (2N, C) -> (2N, 1, C)
        x = x.unsqueeze(1)
        # Reshape y: (2N, C) -> (1, C, 2N)
        y = y.T.unsqueeze(0)
        # Similarity shape: (2N, 2N)
        similarity = th.tensordot(x, y, dims=2)
        return similarity

    def _cosine_simililarity(self, x, y):
        similarity = th.nn.CosineSimilarity(dim=-1)
        # Reshape x: (2N, C) -> (2N, 1, C)
        x = x.unsqueeze(1)
        # Reshape y: (2N, C) -> (1, C, 2N)
        y = y.unsqueeze(0)
        # Similarity shape: (2N, 2N)
        return similarity(x, y)

    def _get_mask_for_neg_samples(self):
        # Diagonal 2Nx2N identity matrix, which consists of four (NxN) quadrants
        diagonal = np.eye(2 * self.batch_size)
        # Diagonal 2Nx2N matrix with 1st quadrant being identity matrix
        q1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        # Diagonal 2Nx2N matrix with 3rd quadrant being identity matrix
        q3 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        # Generate mask with diagonals of all four quadrants being 1.
        mask = th.from_numpy((diagonal + q1 + q3))
        # Reverse the mask: 1s become 0, 0s become 1. This mask will be used to select negative samples
        mask = (1 - mask).type(th.bool)
        # Transfer the mask to the device and return
        return mask.to(self.device)

    def forward(self, representation):
        # Compute similarity matrix
        similarity = self.similarity_fn(representation, representation)
        # Get similarity scores for the positive samples from the diagonal of the first quadrant in 2Nx2N matrix
        l_pos = th.diag(similarity, self.batch_size)
        # Get similarity scores for the positive samples from the diagonal of the third quadrant in 2Nx2N matrix
        r_pos = th.diag(similarity, -self.batch_size)
        # Concatenate all positive samples as a 2nx1 column vector
        positives = th.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        # Get similarity scores for the negative samples (samples outside diagonals in 4 quadrants in 2Nx2N matrix)
        negatives = similarity[self.mask_for_neg_samples].view(2 * self.batch_size, -1)
        # Concatenate positive samples as the first column to negative samples array
        logits = th.cat((positives, negatives), dim=1)
        # Normalize logits via temperature
        logits /= self.temperature
        # Labels are all zeros since all positive samples are the 0th column in logits array.
        # So we will select positive samples as numerator in NTXentLoss
        labels = th.zeros(2 * self.batch_size).to(self.device).long()
        # Compute total loss
        loss = self.criterion(logits, labels)
        # Loss per sample
        closs = loss / (2 * self.batch_size)
        return closs


class NTXent(nn.Module):
    def __init__(self, temperature=1.0):
        """NT-Xent loss for contrastive learning using cosine distance as similarity metric as used in [SimCLR](https://arxiv.org/abs/2002.05709).
        Implementation adapted from https://theaisummer.com/simclr/#simclr-loss-implementation

        Args:
            temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """Compute NT-Xent loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch

        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples

        Returns:
            float: loss
        """
        batch_size = z_i.size(0)

        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)).float()
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss
