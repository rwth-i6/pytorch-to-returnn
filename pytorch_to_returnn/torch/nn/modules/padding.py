
from .module import Module

# No need for these (e.g. ConstantPad1d, ReflectionPad1d, ...),
# as they don't have parameters,
# so it's enough to just wrap `torch.nn.functional.pad`.


class ReflectionPad1d(Module):
  pass
