import sys

print(sys.path)
sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")

from .gradient_registers import *
from .image_loader import *
from .property import *
from .property_parser import *
from .run_utils import *
from .utils import *
