import sys

print(sys.path)
sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")

from .am_loss_function import *
from .am_model import *
from .googlenet import *
from .inceptionv3 import *
from .soft_f1_loss import *
