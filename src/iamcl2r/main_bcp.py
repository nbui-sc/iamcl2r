import argparse
import copy
import datetime
import logging
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import yaml
from dotenv import load_dotenv

from torch.utils.data import DataLoader

import wandb
from iamcl2r.dataset import BalancedBatchSampler, create_data_and_transforms
from iamcl2r.eval import evaluate
from iamcl2r.logger import setup_logger
from iamcl2r.methods import BCPLoss, HocLoss, set_method_configs, info_nce_loss
from iamcl2r.models import create_model
from iamcl2r.params import ExperimentParams
from iamcl2r.train import (retrieval_acc, train_one_bcp_epoch, accuracy)
from iamcl2r.utils import (AverageMeter, broadcast_object, check_params,
                           init_distributed_device, is_master, l2_norm,
                           log_epoch, save_checkpoint)

logger = None


def load_dataloaders(args, scenario_train, scenario_val, task_id, memory, device):
    train_task_set = scenario_train[task_id]
    val_dataset = scenario_val[: task_id + 1]

    batchsampler = None
    batch_size = args.batch_size
    if task_id > 0 and args.scenario == "class-incremental":
        if args.rehearsal > 0:
            mem_x, mem_y, mem_t = memory.get()
            train_task_set.add_samples(mem_x, mem_y, mem_t)
            logger.info(
                f"Memory with {len(mem_x)} examples is added to the training set"
            )
        batchsampler = BalancedBatchSampler(
            train_task_set,
            n_classes=train_task_set.nb_classes,
            batch_size=args.batch_size,
            n_samples=len(train_task_set._x),
            seen_classes=args.seen_classes,
            rehearsal=args.rehearsal,
        )
        train_loader = DataLoader(
            train_task_set, batch_sampler=batchsampler, num_workers=args.num_workers
        )
    else:
        if (
            args.use_subsampled_dataset
            and args.img_per_class * args.classes_at_task[0] < args.batch_size
        ):
            batch_size = args.img_per_class * args.classes_at_task[0]
            logger.info(f"Original batch size of {batch_size} is too high.")
            logger.info(
                f"In current task there are {args.classes_at_task[0]} classes and the dataset has {args.img_per_class} img per class."
            )
            logger.info(f"Setting batch to {batch_size} images per class")
        train_loader = DataLoader(
            train_task_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader


def load_models(
    args,
    current_net_path,
    current_net_backbone,
    previous_net_path,
    previous_net_backbone,
    task_id,
    device,
):
    previous_net = None
    if args.create_old_model:
        logger.info("Creating old model...")
        previous_net = create_model(
            args,
            device=device,
            resume_path=previous_net_path,
            new_classes=0,  # not expanding classifier for old model
            feat_size=args.feat_size,
            backbone=previous_net_backbone,
            n_backward_vers=task_id - 1,
        )
        # set false to require grad for all parameters
        for param in previous_net.parameters():
            param.requires_grad = False
        previous_net.eval()
        logger.info(f"Loaded old model from {current_net_path}")

    net = create_model(
        args,
        device=device,
        resume_path=current_net_path,
        feat_size=args.feat_size,
        backbone=current_net_backbone,
        n_backward_vers=task_id - 1,
    )
    logger.info(f"Created new model from {current_net_path}")

    return net, previous_net


def vanilla_train(
    args,
    device,
    scenario_train,
    scenario_val,
    memory,
    target_transform,
):
    global logger
    logger.info(f"train new model...")

    args.current_backbone = None
    args.seen_classes = []

    for task_id, (train_task_set, _) in enumerate(zip(scenario_train, scenario_val)):
        args.current_task_id = task_id

        if task_id in args.replace_ids:
            resume_path = args.pretrained_model_path[args.replace_ids.index(task_id)]
            args.current_backbone = args.pretrained_backbones[
                args.replace_ids.index(task_id)
            ]
        else:
            resume_path = osp.join(*(args.checkpoint_path, f"ckpt_{task_id-1}.pt"))

        train_loader, val_loader = load_dataloaders(
            args, scenario_train, scenario_val, task_id, memory, device
        )

        net, previous_net = load_models(
            args,
            resume_path,
            args.current_backbone,
            resume_path,
            args.current_backbone,
            task_id,
            device,
        )

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n
        include = lambda n, p: not exclude(n, p)
        named_parameters = list(net.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad
        ]
        rest_params = [
            p for n, p in named_parameters if include(n, p) and p.requires_grad
        ]

        optimizer = optim.SGD(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            lr=args.lr,
            momentum=args.momentum,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
        criterion_cls = nn.CrossEntropyLoss().to(device)

        best_acc = 0
        logger.info(
            f"starting epoch loop at task {task_id+1}/{scenario_train.nb_tasks}"
        )
        init_epoch = 0
        for epoch in range(init_epoch, args.epochs):
            args.current_epoch = epoch
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

            # warmup for the first 10 epochs
            if epoch > 10:
                scheduler_lr.step()

            if (epoch + 1) % args.eval_period == 0 or (epoch + 1) == args.epochs:
                acc_val = retrieval_acc(
                    args,
                    device,
                    net,
                    net,
                    val_loader,
                    val_loader,
                    n_backward_steps=0,
                    identical=True,
                )
                if args.is_main_process:
                    wandb.log({"val/val_acc": acc_val})

                if (acc_val >= best_acc and args.save_best) or (
                    (epoch + 1) == args.epochs and not args.save_best
                ):
                    best_acc = acc_val
                    if args.is_main_process:
                        wandb.log({"val/best_acc": best_acc})
                        save_checkpoint(
                            args, net, optimizer, best_acc, scheduler_lr, backup=False
                        )

            if (
                (epoch + 1) % args.save_period == 0 or (epoch + 1) == args.epochs
            ) and args.is_main_process:
                save_checkpoint(
                    args, net, optimizer, best_acc, scheduler_lr, backup=True
                )

        ## after training in current task
        if args.rehearsal > 0:
            memory.add(*scenario_train[task_id].get_raw_samples(), z=None)
            args.seen_classes = torch.tensor(list(memory.seen_classes), device=device)
            if args.is_main_process:
                # save the memory after new data is added
                memory.save(path=osp.join(args.checkpoint_path, "memory.npz"))
                logger.info(
                    f"memory saved in {osp.join(args.checkpoint_path, 'memory.npz')}"
                )

        if args.distributed:
            dist.barrier()


def post_hoc_align(
    args, device, scenario_train, scenario_val, memory, target_transform
):
    global logger
    logger.info(f"Run post-hoc alignment training...")
    args.current_backbone = None
    args.seen_classes = []

    for task_id, (train_task_set, _) in enumerate(zip(scenario_train, scenario_val)):
        train_loader, val_loader = load_dataloaders(
            args, scenario_train, scenario_val, task_id, memory, device
        )

        # for debugging purposes
        # if task_id < 2:
        #     if args.rehearsal > 0:
        #         memory.add(*scenario_train[task_id].get_raw_samples(), z=None)
        #         args.seen_classes = torch.tensor(list(memory.seen_classes), device=device)
        #     continue

        logger.info("Start training the projection head..")

        if task_id == 0:
            previous_net_path = args.pretrained_model_path[
                args.replace_ids.index(task_id)
            ]
            previous_backbone = args.pretrained_backbones[
                args.replace_ids.index(task_id)
            ]
            current_net_path = osp.join(*(args.checkpoint_path, f"ckpt_{task_id}.pt"))
            current_net_backbone = (
                "from_ckpt"  # automatically set based on the checkpoint
            )
        else:
            previous_net_path = osp.join(
                *(args.checkpoint_path, f"ckpt_{task_id-1}_aligned.pt")
            )
            previous_backbone = "from_ckpt"  # automatically set based on the checkpoint
            current_net_path = osp.join(*(args.checkpoint_path, f"ckpt_{task_id}.pt"))
            current_net_backbone = (
                "from_ckpt"  # automatically set based on the checkpoint
            )

        net, previous_net = load_models(
            args,
            current_net_path,
            current_net_backbone,
            previous_net_path,
            previous_backbone,
            task_id,
            device,
        )
        args.current_backbone = net.backbone_name

        # transfer bc projection head from previous model to current model
        net.bc_projs.load_state_dict(previous_net.bc_projs.state_dict())
        prev_proj_weights = [
            # copy the weights of the previous model
            m.weight.data.clone()
            for m in previous_net.bc_projs
        ]

        # freeze net
        for param in net.parameters():
            param.requires_grad = False
        # add new backward projection layer
        net.add_bc_projection(require_grad=True, proj_type=args.proj_type)

        # get trainable params
        named_parameters = list(net.named_parameters())
        trainable_param = [p for n, p in named_parameters if p.requires_grad]
        for n, p in named_parameters:
            if p.requires_grad:
                logger.info(f"Trainable param: {n}")

        optimizer = optim.SGD(
            [
                {"params": trainable_param, "weight_decay": 0},
            ],
            lr=args.alignment_lr,
            momentum=args.momentum,
        )

        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.alignment_epochs, eta_min=args.min_lr
        )
        criterion_cls = BCPLoss(
            temperature=args.alignment_temperature,
            loss_type=args.alignment_loss_type,
            prev_proj_weights=prev_proj_weights,
        ).to(device)
        init_epoch = 0
        best_acc = 0
        net.to(device)

        for epoch in range(init_epoch, args.alignment_epochs):
            args.current_epoch = epoch
            start = time.time()
            acc_meter = AverageMeter()
            loss_meter = AverageMeter()

            net.train()
            for bid, batchdata in enumerate(train_loader):

                inputs = batchdata[0].to(device, non_blocking=True)
                targets = batchdata[1].to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=args.amp
                ):
                    inputs = inputs[:, 0, :, :, :]
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
                #     n_backward_steps=1,
                # )
                acc_training = 0
                acc_meter.update(acc_training, inputs.size(0))

            # warmup for the first 10 epochs
            if epoch > 10:
                scheduler_lr.step()

            # log after epoch
            lr = optimizer.param_groups[0]["lr"]
            if args.is_main_process:
                wandb.log({"bc_train/bc_epoch": epoch})
                wandb.log({"bc_train/bc_train_loss": loss_meter.avg})
                wandb.log({"bc_train/bc_train_acc": acc_meter.avg})
                wandb.log({"bc_train/alignment_lr": lr})

            end = time.time()
            log_epoch(
                args.alignment_epochs,
                loss_meter.avg,
                acc_meter.avg,
                epoch=epoch,
                task=task_id,
                lr=lr,
                time=end - start,
            )

            if (epoch + 1) % args.eval_period == 0 or (epoch + 1) == args.epochs:
                acc_val = retrieval_acc(
                    args,
                    device,
                    net,
                    previous_net,
                    val_loader,
                    val_loader,
                    n_backward_steps=1,
                )
                if args.is_main_process:
                    wandb.log({"bcval/bc_val_acc": acc_val})

                logger.info(f"BC val acc: {acc_val}, best BC val acc: {best_acc}")

                if (acc_val >= best_acc and args.save_best) or (
                    (epoch + 1) == args.alignment_epochs and not args.save_best
                ):
                    best_acc = acc_val
                    if args.is_main_process:
                        wandb.log({"bcval/bc_best_acc": best_acc})
                        logger.info(f"Best BC val acc: {best_acc}")
                        save_checkpoint(
                            args,
                            net,
                            optimizer,
                            best_acc,
                            scheduler_lr,
                            backup=False,
                            bc=True,
                        )

            if (
                (epoch + 1) % args.save_period == 0
                or (epoch + 1) == args.alignment_epochs
            ) and args.is_main_process:
                save_checkpoint(
                    args, net, optimizer, best_acc, scheduler_lr, backup=True, bc=True
                )

        ## after training in current task
        if args.rehearsal > 0:
            memory.add(*scenario_train[task_id].get_raw_samples(), z=None)
            args.seen_classes = torch.tensor(list(memory.seen_classes), device=device)
            if args.is_main_process:
                # save the memory after new data is added
                memory.save(path=osp.join(args.checkpoint_path, "memory.npz"))
                logger.info(
                    f"Memory saved in {osp.join(args.checkpoint_path, 'memory.npz')}"
                )

        if args.distributed:
            dist.barrier()


def train_and_align(
    args,
    device,
    scenario_train,
    scenario_val,
    memory,
    target_transform,
):
    global logger
    logger.info(f"train and align new model...")

    args.current_backbone = None
    args.seen_classes = []

    for task_id, (train_task_set, _) in enumerate(zip(scenario_train, scenario_val)):
        args.current_task_id = task_id

        if task_id in args.replace_ids:
            resume_path = args.pretrained_model_path[args.replace_ids.index(task_id)]
            args.current_backbone = args.pretrained_backbones[
                args.replace_ids.index(task_id)
            ]
        else:
            resume_path = osp.join(*(args.checkpoint_path, f"ckpt_{task_id-1}.pt"))

        train_loader, val_loader = load_dataloaders(
            args, scenario_train, scenario_val, task_id, memory, device
        )

        net, previous_net = load_models(
            args,
            resume_path,
            args.current_backbone,
            resume_path,
            args.current_backbone,
            task_id,
            device,
        )

        # transfer bc projection head from previous model to current model
        net.bc_projs.load_state_dict(previous_net.bc_projs.state_dict())
        prev_proj_weights = [
            # copy the weights of the previous model
            m.weight.data.clone() for m in previous_net.bc_projs
        ]
        net.add_bc_projection(require_grad=True, proj_type=args.proj_type)

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n
        include = lambda n, p: not exclude(n, p)
        named_parameters = list(net.named_parameters())

        backbone_gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad and "bc_projs" not in n
        ]
        backbone_rest_params = [
            p for n, p in named_parameters if include(n, p) and p.requires_grad and "bc_projs" not in n
        ]
        proj_params = [
            p for n, p in named_parameters if p.requires_grad and "bc_projs" in n
        ]

        optimizer = optim.SGD(
            [
                {"params": backbone_gain_or_bias_params, "weight_decay": 0.0},
                {"params": backbone_rest_params, "weight_decay": args.weight_decay},
                {"params": proj_params, "lr": args.alignment_lr},
            ],
            lr=args.lr,
            momentum=args.momentum,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
        criterion_cls = nn.CrossEntropyLoss().to(device)
        alignment_cls = BCPLoss(
            temperature=args.alignment_temperature,
            loss_type=args.alignment_loss_type,
            prev_proj_weights=prev_proj_weights,
        ).to(device)
        net.to(device)

        best_acc = 0
        logger.info(
            f"starting epoch loop at task {task_id+1}/{scenario_train.nb_tasks}"
        )
        init_epoch = 0
        for epoch in range(init_epoch, args.epochs):
            args.current_epoch = epoch
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
                    assert target_transform is not None, "target_transform is none"
                    # transform targets to write for the end of the feature vector
                    targets = target_transform(targets)
                        
                optimizer.zero_grad()

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                    inputs = inputs.view(-1, c, h, w)
                    # two positive pairs will be near each other in the batch (i and i+1)
                    output = net.bc_forward(inputs, return_original=True, n_backward_steps=1)
                    logits, labels = info_nce_loss(output["original_features"], targets, batch_size, n_views, args.temperature, device)

                    loss = criterion_cls(logits, labels)
                
                    if previous_net is not None:
                        with torch.no_grad():
                            feature_old = previous_net(inputs)["features"]
                        loss = loss + args.alignment_lambda * alignment_cls(output["features"], feature_old, targets)
                
                # loss.backward()
                # optimizer.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_meter.update(loss.item(), inputs.size(0))

                acc_training = accuracy(logits, labels, topk=(1,))[0].item()
                acc_meter.update(acc_training, inputs.size(0))
            
            # log after epoch
            lr = optimizer.param_groups[0]['lr']
            if args.is_main_process:
                wandb.log({'train/epoch': epoch})
                wandb.log({'train/train_loss': loss_meter.avg})
                wandb.log({'train/train_acc': acc_meter.avg})
                wandb.log({'train/lr': lr})

            end = time.time()
            log_epoch(args.epochs, loss_meter.avg, acc_meter.avg, epoch=epoch, task=task_id, lr=lr, time=end-start)

            # warmup for the first 10 epochs
            if epoch > 10:
                scheduler_lr.step()

            if (epoch + 1) % args.eval_period == 0 or (epoch + 1) == args.epochs:
                acc_val = retrieval_acc(
                    args,
                    device,
                    net,
                    net,
                    val_loader,
                    val_loader,
                    n_backward_steps=0,
                )
                if args.is_main_process:
                    wandb.log({"val/val_acc": acc_val})

                if (acc_val >= best_acc and args.save_best) or (
                    (epoch + 1) == args.epochs and not args.save_best
                ):
                    best_acc = acc_val
                    if args.is_main_process:
                        wandb.log({"val/best_acc": best_acc})
                        save_checkpoint(
                            args, net, optimizer, best_acc, scheduler_lr, backup=False
                        )

            if (
                (epoch + 1) % args.save_period == 0 or (epoch + 1) == args.epochs
            ) and args.is_main_process:
                save_checkpoint(
                    args, net, optimizer, best_acc, scheduler_lr, backup=True
                )

        ## after training in current task
        if args.rehearsal > 0:
            memory.add(*scenario_train[task_id].get_raw_samples(), z=None)
            args.seen_classes = torch.tensor(list(memory.seen_classes), device=device)
            if args.is_main_process:
                # save the memory after new data is added
                memory.save(path=osp.join(args.checkpoint_path, "memory.npz"))
                logger.info(
                    f"memory saved in {osp.join(args.checkpoint_path, 'memory.npz')}"
                )

        if args.distributed:
            dist.barrier()



def main():
    # load params from the config file from yaml to dataclass
    parser = argparse.ArgumentParser(
        description='Official PyTorch Implementation of "Stationary Representations: Optimally Approximating Compatibility and Implications for Improved Model Replacements" CVPR24'
    )
    parser.add_argument(
        "-c", "--config_path", help="path of the experiment yaml", default=osp.join(os.getcwd(), "configs/hoc.yaml"), type=str,
    )
    parser.add_argument(
        "-e", "--eval_only", help="only evaluation", action="store_true"
    )
    parser.add_argument(
        "-d", "--debug", help="only evaluation", action="store_true"
    )

    params = parser.parse_args()
    base_config_name = osp.splitext(osp.basename(params.config_path))[0]
    if params.debug:
        wandb.init(mode="disabled")

    with open(params.config_path, "r") as stream:
        loaded_params = yaml.safe_load(stream)
    args = ExperimentParams()

    for k, v in loaded_params.items():
        args.__setattr__(k, v)

    args.yaml_name = os.path.basename(params.config_path)

    args.eval_only = args.eval_only or params.eval_only
    # reproducibility
    args.seed = np.random.randint(0, 10000) if args.seed == 0 else args.seed

    device = init_distributed_device(args)
    args.is_main_process = is_master(args)

    if not osp.exists(args.data_path) and args.is_main_process:
        os.makedirs(args.data_path)


    if not args.eval_only or args.checkpoint_path is None:
        checkpoint_path = osp.join(*(args.output_folder, f"{base_config_name}"))
        if args.distributed:
            checkpoint_path = broadcast_object(args, checkpoint_path)
        args.checkpoint_path = checkpoint_path
        if not osp.exists(args.checkpoint_path) and args.is_main_process:
            os.makedirs(args.checkpoint_path)

    log_file = (
        f"train-{datetime.datetime.now().strftime('%H%M%S')}-gpu{device.index}.log"
        if not args.eval_only
        else f"eval.log"
    )
    setup_logger(
        logfile=os.path.join(*(args.checkpoint_path, log_file)),
        console_log=args.is_main_process,
        file_log=True,
        log_level="INFO",
    )

    global logger
    if args.is_main_process:
        run = wandb.init(dir=".", tags=[])
    if not args.eval_only:
        logger = logging.getLogger("IAM-CL2R-Train")
    else:
        logger = logging.getLogger("IAM-CL2R-Eval")
        run.tags = run.tags + ("eval",)

    set_method_configs(args, name=args.method)

    # reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.replace_ids += [0]
    args.replace_ids.sort()

    set_method_configs(args, name=args.method)

    check_params(args)

    if args.is_main_process:
        wandb.config.update(vars(args))

    logger.info(f"Running mode: {'evaluation' if args.eval_only else 'training'} {'+ debug' if params.debug else ''}")
    logger.info(f"Current args:\n{vars(args)}")
    logger.info(f"data of this run is stored in this path: {args.checkpoint_path}")
    logger.info(f"Logging on device {device}.")

    is_post_hoc = (args.alignment_strategy == "post-hoc")

    data = create_data_and_transforms(args)
    scenario_train = data["scenario_train"]
    scenario_val = data["scenario_val"]
    memory = data["memory"]
    new_memory = copy.deepcopy(memory)
    target_transform = data["target_transform"]

    if not args.eval_only:
        if is_post_hoc:
            vanilla_train(args, device, scenario_train, scenario_val, memory, target_transform)
            post_hoc_align(args, device, scenario_train, scenario_val, new_memory, target_transform)
        else:
            train_and_align(args, device, scenario_train, scenario_val, memory, target_transform)

    if not args.train_only:
        logger.info(f"Starting Evaluation")
        ntasks_eval = args.ntasks_eval if args.ntasks_eval else args.nb_tasks_evaluation
        if args.checkpoint_path is None:
            args.checkpoint_path = osp.join(*(args.output_folder, f"{base_config_name}"))
            assert osp.exists(args.checkpoint_path), f"Checkpoint path {args.checkpoint_path} does not exist"
        data = create_data_and_transforms(args, mode="identification")
        query_loader = data["query_loader"]
        gallery_loader = data["gallery_loader"]
        evaluate(args, device, query_loader, gallery_loader, ntasks_eval=ntasks_eval, is_post_hoc=True)

        if args.is_main_process:
            artifact = wandb.Artifact("compatibility-matrix", type="text")
            artifact.add_file(osp.join(*(args.checkpoint_path, "comp-matrix.txt")))
            run.log_artifact(artifact)
            wandb.finish()

    return 0


if __name__ == "__main__":
    load_dotenv()
    main()
