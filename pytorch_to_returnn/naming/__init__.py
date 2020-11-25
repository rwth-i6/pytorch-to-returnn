
from .call import CallEntry
from .module import ModuleEntry
from .namespace import RegisteredName
from .naming import Naming
from .returnn_ctx import ReturnnContext
from .tensor import TensorEntry
import os as _os

__doc__ = open(_os.path.dirname(__file__) + "/README.rst").read()
