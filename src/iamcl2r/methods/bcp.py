import torch
import torch.nn as nn


__configs = {
    'name': 'bcp',
    'create_old_model': True,
    'preallocated_classes': 1024,
    'mu_': 10,
}


def BCPconfigs():
    return __configs


def CLRBCPconfigs():
    return {
        'name': 'clr_hcp',
        'create_old_model': True,
        'mu_': 10,
    }


class BCPLoss(nn.Module):
    def __init__(self, mu_, loss_type='nce'):
        super(BCPLoss, self).__init__()
        self.loss_type = loss_type
        self.mu_ = mu_
    
    def forward(self, 
                feat_new, 
                feat_old, 
                labels, 
               ):
        if self.loss_type == 'nce':
            loss = self.nce_loss(feat_new, feat_old, labels)
        elif self.loss_type == 'mse':
            loss = self.mse_loss(feat_new, feat_old)
        return loss

    def mse_loss(self, feat_new, feat_old):
        """Calculates MSE loss.
        This
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
        return torch.nn.functional.mse_loss(feat_new, feat_old)

    def nce_loss(self, feat_new, feat_old, labels):
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
        batch_size = feat_old.shape[0]
        diag_mask = torch.eye(batch_size, device=feat_old.device, dtype=torch.bool)
        sim_01 = torch.einsum("nc,mc->nm", feat_old, feat_new) *  self.mu_

        positive_loss = -sim_01[diag_mask]
        # Get the labels of out0 and out1 samples
        labels_0 = labels.unsqueeze(1).expand(-1, batch_size)  # Shape: (batch_size, batch_size)
        labels_1 = labels.unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, batch_size)

        # Mask similarities between the same class
        class_mask = labels_0 == labels_1
        sim_01 = (sim_01* (~class_mask)).view(batch_size, -1)

        negative_loss_01 = torch.logsumexp(sim_01, dim=1)
        return (positive_loss + negative_loss_01).mean()