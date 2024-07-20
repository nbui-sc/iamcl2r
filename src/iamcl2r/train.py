import numpy as np
import torch
import torch.nn.functional as F

import time
import wandb


from iamcl2r.models import extract_features
from iamcl2r.performance_metrics import identification
from iamcl2r.utils import AverageMeter, log_epoch, l2_norm


def train_one_epoch(args,
                    device, 
                    net, 
                    previous_net, 
                    train_loader, 
                    scaler,
                    optimizer,
                    epoch, 
                    criterion_cls, 
                    task_id, 
                    add_loss,
                    target_transform=None
                   ):
    start = time.time()
    
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    net.train()
    for bid, batchdata in enumerate(train_loader):
        
        inputs = batchdata[0].to(device, non_blocking=True) 
        targets = batchdata[1].to(device, non_blocking=True)  

        if args.fixed:
            assert target_transform is not None, "target_transform is None"
            # transform targets to write for the end of the feature vector
            targets = target_transform(targets)
                
        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
            output = net(inputs)
            loss = criterion_cls(output["logits"], targets)
        
            if previous_net is not None:
                with torch.no_grad():
                    feature_old = previous_net(inputs)["features"]
                if "hoc" in args.method:
                    norm_feature_old = l2_norm(feature_old)
                    norm_feature_new = l2_norm(output["features"])
                    loss_feat = add_loss(norm_feature_new, norm_feature_old, targets)
                    # Eq. 3 in the paper
                    loss = loss * args.lambda_ + (1 - args.lambda_) * loss_feat
                else:
                    raise NotImplementedError(f"Method {args.method} not implemented")
        
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), inputs.size(0))

        acc_training = accuracy(output["logits"], targets, topk=(1,))
        acc_meter.update(acc_training[0].item(), inputs.size(0))
    
    # log after epoch
    if args.is_main_process:
        wandb.log({'train/epoch': epoch})
        wandb.log({'train/train_loss': loss_meter.avg})
        wandb.log({'train/train_acc': acc_meter.avg})
        wandb.log({'train/lr': optimizer.param_groups[0]['lr']})

    end = time.time()
    log_epoch(args.epochs, loss_meter.avg, acc_meter.avg, epoch=epoch, task=task_id, time=end-start)


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
    

def classification(args, device, net, loader, criterion_cls, target_transform=None):
    classification_loss_meter = AverageMeter()
    classification_acc_meter = AverageMeter()
    
    net.eval()
    with torch.no_grad():
        for bid, batchdata in enumerate(loader):
        
            inputs = batchdata[0].to(device, non_blocking=True) 
            targets = batchdata[1].to(device, non_blocking=True) 
            if args.fixed:
                # transform targets to write for the end of the interface vector
                targets = target_transform(targets)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                output = net(inputs)
                assert torch.all(targets >= 0) and torch.all(targets <= output["logits"].size(1))

                loss = criterion_cls(output["logits"], targets)

            classification_acc = accuracy(output["logits"], targets, topk=(1,))
            
            classification_loss_meter.update(loss.item(), inputs.size(0))
            classification_acc_meter.update(classification_acc[0].item(), inputs.size(0))

    log_epoch(loss=classification_loss_meter.avg, acc=classification_acc_meter.avg, classification=True)

    classification_acc = classification_acc_meter.avg

    return classification_acc


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


def train_one_clr_epoch(
    args,
    device,
    net,
    previous_net,
    train_loader,
    scaler,
    optimizer,
    epoch,
    criterion_cls,
    task_id,
    add_loss,
    target_transform=None
):
    start = time.time()

    acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    net.train()
    for bid, batchdata in enumerate(train_loader):
        
        inputs = batchdata[0].to(device, non_blocking=True) 
        batch_size, n_views, c, h, w = inputs.size()

        targets = batchdata[1].to(device, non_blocking=True)  
        targets = torch.stack([targets for _ in range(n_views)], dim=1).view(-1)

        if args.fixed:
            assert target_transform is not None, "target_transform is None"
            # transform targets to write for the end of the feature vector
            targets = target_transform(targets)
                
        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
            inputs = inputs.view(-1, c, h, w)
            # two positive pairs will be near each other in the batch (i and i+1)
            output = net(inputs)
            logits, labels = info_nce_loss(output["features"], targets, batch_size, n_views, args.temperature, device)

            loss = criterion_cls(logits, labels)
        
            if previous_net is not None:
                with torch.no_grad():
                    feature_old = previous_net(inputs)["features"]
                if 'hoc' in args.method:
                    assert add_loss is not None, "add_loss is None"
                    norm_feature_old = l2_norm(feature_old)
                    norm_feature_new = l2_norm(output["features"])
                    loss_feat = add_loss(norm_feature_new, norm_feature_old, targets)
                    # Eq. 3 in the paper
                    loss = loss * args.lambda_ + (1 - args.lambda_) * loss_feat
                elif 'bcp' in args.method:
                    pass
                else:
                    raise NotImplementedError(f"Method {args.method} not implemented")
        
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), inputs.size(0))

        acc_training = accuracy(logits, labels, topk=(1,))
        acc_meter.update(acc_training[0].item(), inputs.size(0))
    
    # log after epoch
    lr = optimizer.param_groups[0]['lr']
    if args.is_main_process:
        wandb.log({'train/epoch': epoch})
        wandb.log({'train/train_loss': loss_meter.avg})
        wandb.log({'train/train_acc': acc_meter.avg})
        wandb.log({'train/lr': lr})

    end = time.time()
    log_epoch(args.epochs, loss_meter.avg, acc_meter.avg, epoch=epoch, task=task_id, lr=lr, time=end-start)

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


def train_one_bcp_epoch(
    args,
    device,
    net,
    previous_net,
    train_loader,
    scaler,
    optimizer,
    epoch,
    criterion_cls,
    task_id,
    target_transform=None
):
    start = time.time()

    acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    net.train()
    for bid, batchdata in enumerate(train_loader):
        
        inputs = batchdata[0].to(device, non_blocking=True) 
        batch_size, n_views, c, h, w = inputs.size()

        targets = batchdata[1].to(device, non_blocking=True)  
        targets = torch.stack([targets for _ in range(n_views)], dim=1).view(-1)

        if args.fixed:
            assert target_transform is not None, "target_transform is None"
            # transform targets to write for the end of the feature vector
            targets = target_transform(targets)
                
        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
            inputs = inputs.view(-1, c, h, w)
            # two positive pairs will be near each other in the batch (i and i+1)
            output = net.bc_forward(inputs, n_backward_steps=1)
            features = output["features"]
            with torch.no_grad():
                feature_old = previous_net(inputs)["features"]

            loss = criterion_cls(features, feature_old, targets)
        
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), inputs.size(0))

        # acc_training = retrieval_acc(
        #     args, 
        #     device,
        #     net, 
        #     previous_net,
        #     train_loader,
        #     train_loader,
        # )
        acc_training = 0
        acc_meter.update(acc_training, inputs.size(0))
    
    # log after epoch
    lr = optimizer.param_groups[0]['lr']
    if args.is_main_process:
        wandb.log({'bc_train/bc_epoch': epoch})
        wandb.log({'bc_train/bc_train_loss': loss_meter.avg})
        wandb.log({'bc_train/bc_train_acc': acc_meter.avg})
        wandb.log({'bc_train/bc_lr': lr})

    end = time.time()
    log_epoch(args.bc_epochs, loss_meter.avg, acc_meter.avg, epoch=epoch, task=task_id, lr=lr, time=end-start)
