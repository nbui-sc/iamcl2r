import torch
import torch.nn.functional as F


def info_nce_loss(
    features,
    targets,
    batch_size,
    n_views,
    temperature,
    device,
):
    labels = torch.stack([torch.arange(batch_size) for _ in range(n_views)], dim=1).view(-1)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # mask out examples of the same class but not augmentations
    # class_mask = 
    class_mask = (targets.unsqueeze(0) == targets.unsqueeze(1)).float()
    similarity_matrix = similarity_matrix * (1 - (torch.logical_xor(class_mask, labels)).float())
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    # print(logits.shape, labels.shape)
    # print(logits, labels)
    # raise ValueError
    return logits, labels
