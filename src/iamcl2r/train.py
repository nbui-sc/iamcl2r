import numpy as np
import torch
import torch.nn.functional as F
import logging
from torch import nn

import time
import wandb


from iamcl2r.models import extract_features
from iamcl2r.performance_metrics import identification
from iamcl2r.utils import AverageMeter, log_epoch, l2_norm

logger = logging.getLogger(__name__)









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
