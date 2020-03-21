import image_loader as il
import voc_loader as vl
from torch.utils.data import DataLoader
import property as P
import sys
import traceback
import am_model as ss
import first_attention_module_train as amt
import argparse
import os
import shutil
import time
import random

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from MODELS.model_resnet import *
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

if __name__ == "__main__":
    # default parser
    parser = P.parse_input_commands()

    # extend parser
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--att-type', type=str, choices=['BAM', 'CBAM'], default=None)
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
    #                     help='model architecture: ' +
    #                          ' | '.join(model_names) +
    #                          ' (default: resnet18)')
    parser.add_argument('--depth', default=50, type=int, metavar='D',
                        help='model depth')

    parsed = parser.parse_args(sys.argv[1:])
    gpu = int(parsed.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    gpu = 0
    parsed_description = parsed.description
    pre_train = int(parsed.pre_train)
    train_set_size = int(parsed.train_set)
    epochs = int(parsed.epochs)
    run_name = parsed.run_name
    algorithm_name = parsed.algorithm_name

    description = "description-{},train_set-{},epochs-{}".format(
        parsed_description,
        train_set_size,
        epochs
    )

    P.initialize_log_name(run_name, algorithm_name, description)

    P.write_to_log("description={}".format(description))
    P.write_to_log("classes={}".format(classes))
    P.write_to_log("run=" + run_name)
    P.write_to_log("algorithm_name=" + algorithm_name)

    log_name, log_dir = os.path.basename(P.log)[:-4], os.path.dirname(P.log)

    snapshots_path = os.path.join(log_dir, log_name)
    os.makedirs(snapshots_path, exist_ok=True)

    model = ResidualNet('ImageNet', parsed.depth, classes, parsed.att_type)
    P.write_to_log(model)

    v = vl.VocDataLoader(P.voc_data_path, ['bus'], [6])
    train_data_set = DataLoader(vl.VocDataset(v.train_data), batch_size=5, shuffle=True)
    test_data_set = DataLoader(vl.VocDataset(v.test_data), batch_size=5)

