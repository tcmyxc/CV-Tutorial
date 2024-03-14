from functools import partial
import torch.nn as nn

import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from loss.fl import focal_loss
from loss.cal_loss import cal_loss


def get_loss_fn(args, **kwargs):
    if args.loss_type == "ce":
        loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.loss_type == "fl":
        loss_fn = partial(focal_loss, gamma=2)
    elif args.loss_type == "cal":
        loss_fn = cal_loss
    else:
        raise NotImplementedError(f'{args.loss_type} is not implemented')

    return loss_fn


if __name__ == "__main__":
    print("hello world")