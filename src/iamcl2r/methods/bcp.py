import torch
import torch.nn as nn


__configs = {
    'name': 'bcp',
    'create_old_model': True,
    'temperature': 0.07,
}


def BCPconfigs():
    return __configs


def CLRBCPconfigs():
    return {
        'name': 'clr_hcp',
        'create_old_model': True,
        'temperature': 0.07,
    }


class BCPLoss(nn.Module):
    def __init__(self, temperature, loss_type='nce', prev_proj_weights=None):
        super(BCPLoss, self).__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.prev_proj_weights = prev_proj_weights
    
    def forward(self, 
                feat_new, 
                feat_old, 
                labels,
               ):
        if self.loss_type == 'nce':
            loss = self.nce_loss(feat_new, feat_old, labels)
        elif self.loss_type == 'single-step-mse':
            loss = self.mse_loss(feat_new, feat_old)
        elif self.loss_type == 'multi-step-mse':
            loss = self.multi_step_mse_loss(feat_new, feat_old)
        else:
            raise ValueError(f"Loss type {self.loss_type} not supported.")
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
        return nn.functional.mse_loss(feat_new, feat_old)

    def multi_step_mse_loss(self, feat_new, feat_old):
        """Calculates Joint MSE loss.
        """
        feat_diff = feat_new - feat_old
        loss = torch.mean(feat_diff**2)
        for w in self.prev_proj_weights:
            loss += torch.mean(nn.functional.linear(feat_diff, w)**2)
        return loss

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
        feat_new = nn.functional.normalize(feat_new, p=2, dim=1)
        feat_old = nn.functional.normalize(feat_old, p=2, dim=1)
        ## create diagonal mask that only selects similarities between
        ## representations of the same images
        batch_size = feat_old.shape[0]
        diag_mask = torch.eye(batch_size, device=feat_old.device, dtype=torch.bool)
        sim_01 = torch.einsum("nc,mc->nm", feat_old, feat_new) /  self.temperature

        positive_loss = -sim_01[diag_mask]
        # Get the labels of out0 and out1 samples
        labels_0 = labels.unsqueeze(1).expand(-1, batch_size)  # Shape: (batch_size, batch_size)
        labels_1 = labels.unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, batch_size)

        # Mask similarities between the same class
        class_mask = labels_0 == labels_1
        sim_01 = (sim_01 * (~class_mask)).view(batch_size, -1)

        negative_loss_01 = torch.logsumexp(sim_01, dim=1)
        return (positive_loss + negative_loss_01).mean()
