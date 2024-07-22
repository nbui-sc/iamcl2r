import torch
from torch import nn
from torch.nn import functional as F


class NCEAlignmentLoss(nn.Module):
    def __init__(self, mu_):
        super(NCEAlignmentLoss, self).__init__()
        self.mu_ = mu_
    
    def forward(self, 
                feat_new, 
                feat_old, 
                labels, 
               ):
        feat_old = F.normalize(feat_old, dim=1)
        feat_new = F.normalize(feat_new, dim=1)
        loss = self._loss(feat_old, feat_new, labels)
        return loss

    def _loss(self, out0, out1, labels):
        """Calculates infoNCE loss.
        This code implements Equation 4 in "Stationary Representations: Optimally Approximating Compatibility and Implications for Improved Model Replacements"

        Args:
            feat_old:
                features extracted with the old model.
                Shape: (batch_size, embedding_size)
            feat_new:
                features extracted with the new model.
                Shape: (batch_size, embedding_size)
            labels:
                Labels of the images.
                Shape: (batch_size,)
        Returns:
            Mean loss over the mini-batch.
        """
        ## create diagonal mask that only selects similarities between
        ## representations of the same images
        batch_size = out0.shape[0]
        diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)
        sim_01 = torch.einsum("nc,mc->nm", out0, out1) *  self.mu_

        positive_loss = -sim_01[diag_mask]
        # Get the labels of out0 and out1 samples
        labels_0 = labels.unsqueeze(1).expand(-1, batch_size)  # Shape: (batch_size, batch_size)
        labels_1 = labels.unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, batch_size)

        # Mask similarities between the same class
        class_mask = labels_0 == labels_1
        sim_01 = (sim_01* (~class_mask)).view(batch_size, -1)

        negative_loss_01 = torch.logsumexp(sim_01, dim=1)
        return (positive_loss + negative_loss_01).mean()


class SimCLRLoss(nn.Module):
    def __init__(self, temperature, device):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, 
                feat_new, 
                feat_old, 
                labels, 
               ):
        loss = self._loss(feat_old, feat_new, labels)
        return loss

    @staticmethod
    def info_nce_loss(features, targets, batch_size, n_views):
        labels = torch.stack([torch.arange(batch_size) for _ in range(n_views)], dim=1).view(-1)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # mask out examples of the same class but not augmentations
        class_mask = (targets.unsqueeze(0) == targets.unsqueeze(1)).float()
        similarity_matrix = similarity_matrix * (1 - (torch.logical_xor(class_mask, labels)).float())
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / self.temperature
        # print(logits.shape, labels.shape)
        # print(logits, labels)
        # raise ValueError
        return logits, labels
