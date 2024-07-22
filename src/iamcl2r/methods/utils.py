import torch
import torch.nn as nn
import torch.nn.functional as F
from iamcl2r.models import extract_features
from iamcl2r.performance_metrics import identification


def update_config(args, method_configs):
    for k, v in method_configs:
        args.__setattr__(k, v)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def retrieval_acc(args, device, net, previous_net, query_loader, gallery_loader, n_backward_steps=0):
    net.eval() 
    query_feat, targets = extract_features(
        args,
        device,
        net,
        query_loader,
        return_labels=True,
        n_backward_steps=n_backward_steps
    )
    
    gallery_feat, gallery_targets = extract_features(
        args,
        device,
        previous_net,
        gallery_loader,
        return_labels=True,
        n_backward_steps=0,
    )

    acc = identification(gallery_feat, gallery_targets, 
                         query_feat, targets, 
                         topk=1)
    return acc
