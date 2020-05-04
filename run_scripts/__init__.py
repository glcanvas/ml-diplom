import sys

print(sys.path)
sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")

from .executor_nvsmi import *
from .initial_strategies import *
from .local_executor import *
from .on_server_executor import *
