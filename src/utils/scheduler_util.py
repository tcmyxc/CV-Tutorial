"""
学习率调度器
"""
import torch


def get_lr_scheduler(optimizer, args):
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    elif args.lr_scheduler == "multisteplr":
        if "cifar" in args.data_name:
            if args.epochs == 200:
                cifar_milestones = [60, 120, 160]
                cifar_milestones = [(i - args.lr_warmup_epochs) for i in cifar_milestones]
                main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cifar_milestones, gamma=0.2)
            elif args.epochs == 300:
                cifar_milestones = [150, 225]
                cifar_milestones = [(i - args.lr_warmup_epochs) for i in cifar_milestones]
                main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cifar_milestones, gamma=0.1)
        elif args.data_name == "svhn":
            svhn_milestones = [80, 120]
            svhn_milestones = [(i - args.lr_warmup_epochs) for i in svhn_milestones]
            main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=svhn_milestones, gamma=0.1)
        else:
            raise RuntimeError(f"Invalid lr scheduler '{args.lr_scheduler}'")
    else:
        raise RuntimeError(f"Invalid lr scheduler '{args.lr_scheduler}'")

    # 学习率预热
    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    return lr_scheduler
