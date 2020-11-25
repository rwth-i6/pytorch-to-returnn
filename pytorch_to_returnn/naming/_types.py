
"""
Only provides types used for type checking, and auto-completion, etc.

Note that this `naming` submodule can be used for different `torch` variants
(e.g. traced original, or our Torch-RETURNN wrapper).
"""

import typing

if typing.TYPE_CHECKING:
  from pytorch_to_returnn.torch.tensor import Tensor
  from pytorch_to_returnn.torch.nn import Module

else:

  class Module:  # Dummy
    pass

  class Tensor:  # Dummy
    pass
