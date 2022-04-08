
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

int = dtype("int32")
int32 = dtype("int32")
long = dtype("int64")
int64 = dtype("int64")
half = dtype("float16")
float = dtype("float32")
float32 = dtype("float32")
double = dtype("float64")
float64 = dtype("float64")
