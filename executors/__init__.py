import sys

print(sys.path)
sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")
sys.path.insert(0, "/home/ubuntu/ml3/ml-diplom")

from .abastract_executor import *
from .executor_baseline_vgg16 import *
from .executor_inceptionv import *
from .executor_resnet import *
from .executor_sequential import *
from .executor_simultaneous import *
