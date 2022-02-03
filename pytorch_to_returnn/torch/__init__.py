
from .tensor import *
from ._C import *
from .nn.functional import *
from .serialization import load
from .autograd import *
from . import nn
from . import cuda
from . import onnx
from . import jit

__version__ = "1.8.1"
__returnn__ = True
