import torch
import torch.nn as nn

from .train_and_align import train_and_align
from .utils import update_config
from .losses import NCEAlignmentLoss



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
    assert args.fixed is True, "Fixed is not True"
    assert target_transform is not None, "target_transform is None"

    reps_criterion = nn.CrossEntropyLoss().to(device)
    alignment_criterion = NCEAlignmentLoss(args.mu_).to(device)

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
