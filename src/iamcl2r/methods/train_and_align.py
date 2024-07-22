import time
import logging
import wandb
import os.path as osp

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from iamcl2r.utils import AverageMeter, log_epoch, l2_norm
from iamcl2r.utils import save_checkpoint

from .utils import accuracy

logger = logging.getLogger(__name__)


def validate(args, device, net, loader, criterion_cls, target_transform=None):
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

    log_epoch(loss=classification_loss_meter.avg, acc=classification_acc_meter.avg, validation=True)

    classification_acc = classification_acc_meter.avg

    return classification_acc


def train_one_epoch(
    args,
    net, 
    previous_net, 
    train_loader, 
    scaler,
    optimizer,
    scheduler_lr,
    epoch, 
    reps_criterion,
    alignment_criterion,
    device, 
    task_id, 
    target_transform=None,
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
            if len(inputs.size()) == 5:
                b, n, c, h, w = inputs.size()
                inputs = inputs.view(-1, c, h, w)
            output = net(inputs)
            loss = reps_criterion(output["logits"], targets, **kwargs)
        
            if previous_net is not None and alignment_criterion is not None:
                with torch.no_grad():
                    feature_old = previous_net(inputs)["features"]

                loss_feat = alignment_criterion(output["features"], feature_old, targets)
                loss = loss * args.lambda_ + (1 - args.lambda_) * loss_feat
        
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
    logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    log_epoch(args.epochs, loss_meter.avg, acc_meter.avg, epoch=epoch, task=task_id, time=end-start)


def train_and_align(
    args,
    net,
    previous_net,
    train_loader,
    val_loader,
    scenario_train,
    reps_criterion,
    alignment_criterion,
    memory,
    task_id,
    device,
    target_transform=None,
):
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)
    named_parameters = list(net.named_parameters())

    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    optimizer = optim.SGD(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.weight_decay},
        ],
        lr=args.lr, 
        momentum=args.momentum, 
    )

    # backbone_params = [p for n, p in named_parameters if "backbone" in n and p.requires_grad]
    # rest_params = [p for n, p in named_parameters if "backbone" not in n and p.requires_grad]
    
    # optimizer = optim.Adam(
    #                     [
    #                         {"params": backbone_params, "lr": args.backbone_lr},
    #                         {"params": rest_params,},
    #                     ],
    #                     lr=args.lr, 
    #                     weight_decay=args.weight_decay,
    #                     )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    # scheduler_lr = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
    criterion_cls = nn.CrossEntropyLoss().to(device)
        
    best_acc = 0
    init_epoch = 0
    for epoch in range(init_epoch, args.epochs):
        args.current_epoch = epoch
        train_one_epoch(args=args,
                        net=net,
                        previous_net=previous_net,
                        train_loader=train_loader,
                        scaler=scaler,
                        optimizer=optimizer,
                        scheduler_lr=scheduler_lr,
                        epoch=epoch, 
                        reps_criterion=reps_criterion,
                        alignment_criterion=alignment_criterion,
                        device=device,
                        task_id=task_id,
                        target_transform=target_transform,
                        )
        scheduler_lr.step()

        if (epoch + 1) % args.eval_period == 0 or (epoch + 1) == args.epochs:
            acc_val = validate(args,
                               device,
                               net,
                               val_loader,
                               criterion_cls,
                               target_transform=target_transform)
            if args.is_main_process:
                wandb.log({'val/val_acc': acc_val}) 
                                
            if (acc_val >= best_acc and args.save_best) or ((epoch + 1) == args.epochs and not args.save_best):
                best_acc = acc_val
                if args.is_main_process:
                    wandb.log({'val/best_acc': best_acc}) 
                    save_checkpoint(args, net, 
                                    optimizer, best_acc, scheduler_lr, backup=False)

        if ((epoch + 1) % args.save_period == 0 or (epoch + 1) == args.epochs) and args.is_main_process:
            save_checkpoint(args, net, 
                            optimizer, best_acc, scheduler_lr, backup=True)
        
    ## after training in current task
    if args.rehearsal > 0:
        memory.add(*scenario_train[task_id].get_raw_samples(), z=None)
        if args.is_main_process:
            # save the memory after new data is added
            memory.save(path=osp.join(args.checkpoint_path, "memory.npz"))   
            logger.info(f"Memory saved in {osp.join(args.checkpoint_path, 'memory.npz')}")
            save_checkpoint(args, net, optimizer, 
                            best_acc, scheduler_lr, backup=True)

    if args.distributed:
        dist.barrier()
