import sys

print(sys.path)
sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")

from .abstract_train import *
from .sequential_train import *
from .simultaneous_train import *
from .vgg16_baseline_strategy import *
