import torch

from .train_and_align import train_and_align
from .train_and_align_clr import train_and_align_clr
from .losses import SimCLRLoss

from torch import nn


def train(
    args,
    net,
    previous_net,
    train_loader,
    val_loader,
    scenario_train,
    memory,
    task_id,
    device,
    target_transform=None,
):
    reps_criterion = nn.CrossEntropyLoss().to(device)
    alignment_criterion = None

    train_and_align(
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
        target_transform
    )

def train_one_epoch_clr(
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
            loss = reps_criterion(output["logits"], targets, batch_size=b, n_views=n)
        
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




def train_clr(
    args,
    net,
    previous_net,
    train_loader,
    val_loader,
    scenario_train,
    memory,
    task_id,
    device,
    target_transform=None,
):
    reps_criterion = SimCLRLoss(temperature=args.simclr_temperature, device=device)
    alignment_criterion = None

    train_and_align_clr(
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
        target_transform,
    )
