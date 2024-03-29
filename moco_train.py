import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import moco.builder
import moco.loader
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from utils.torch_utils import select_device
from models.mocoBuilder import Model
from moco.utils import AverageMeter, ProgressMeter, adjust_learning_rate, accuracy, save_checkpoint


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch SID RAW Training")
    parser.add_argument("--dataset_path", type=str, default="", help="dataset path")
    parser.add_argument("--pair_list", type=str, default="", help="dataset path")
    parser.add_argument("--epochs", type=int, default=200, help="number of total epochs to run")
    parser.add_argument("--start-epoch", type=int, default=0, help="manual epoch number (useful on restarts)")
    parser.add_argument("--batch-size", type=int, default=8, help="mini batch")
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="momentum of SGD solver")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay (default: 1e-4)",)
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", default=8, type=int, help="number of data loading workers (default: 32)")
    parser.add_argument("--save_path", type=str, default="", help="dataset path")
    args = parser.parse_args()
    return args


def main():
    args = get_parser()
    device = select_device(args.device)

    print(">>> Creating Model <<<")
    model = moco.builder.MoCo(Model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    print(">>> Data Loading <<<")
    train_set = moco.loader.MoCoData(args.dataset_path, args.piar_list)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )

    print(">>> Training <<<")
    for epoch in range(args.start_epoch, args.epoch):
        step_one_train(train_loader, model, criterion, optimizer, epoch, args, device)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            is_best=False,
            filename=os.path.join(args.save_path, "checkpoint_{:04d}.pt".format(epoch))
        )


def step_one_train(train_loader, model, criterion, optimizer, epoch, args, device):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()

    end = time.time()

    for i, images_ in enumerate(train_loader):
        data_time.update(time.time() - end)
        input_q = images_["img_q"].to(device)
        input_k = images_["img_k"].to(device)

        output, target = model(im_q=input_q, im_k=input_k)
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input_q.size(0))
        top1.update(acc1[0], input_q.size(0))
        top5.update(acc5[0], input_q.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == "__main__":
    main()
