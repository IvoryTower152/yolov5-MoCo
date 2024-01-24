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


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch SID RAW Training")
    parser.add_argument("--dataset_path", type=str, default="", help="dataset path")
    parser.add_argument("--epochs", type=int, default=200, help="number of total epochs to run")
    parser.add_argument("--start-epoch", type=int, default=0, help="manual epoch number (useful on restarts)")
    parser.add_argument("--batch-size", type=int, default=8, help="mini batch")
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="momentum of SGD solver")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay (default: 1e-4)",)
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
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


if __name__ == "__main__":
    main()
