from .train_and_align import train_and_align
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
