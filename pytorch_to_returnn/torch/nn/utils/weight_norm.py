

from typing import TypeVar, Any
from .. import Module
from ..parameter import Parameter
from ...tensor import Tensor
from ..functional import norm_except_dim


class WeightNorm(object):
  name: str
  dim: int

  def __init__(self, name: str, dim: int) -> None:
    if dim is None:
      dim = -1
    self.name = name
    self.dim = dim

  def compute_weight(self, module: Module) -> Any:
    g = getattr(module, self.name + '_g')
    v = getattr(module, self.name + '_v')
    # or native _weight_norm
    return v * (g / norm_except_dim(v, 2, self.dim))

  @staticmethod
  def apply(module, name: str, dim: int) -> 'WeightNorm':
    if dim is None:
      dim = -1

    fn = WeightNorm(name, dim)

    weight = getattr(module, name)

    # remove w from parameter list
    del module._parameters[name]

    # add g and v as new parameters and express w as g/||v|| * v
    module.register_parameter(name + '_g', Parameter(norm_except_dim(weight, 2, dim).data))
    module.register_parameter(name + '_v', Parameter(weight.data))
    setattr(module, name, fn.compute_weight(module))

    # The original logic here was:
    #   # recompute weight before every forward()
    #   module.register_forward_pre_hook(fn)
    # However, we don't use TF in eager mode,
    # i.e. there is no need to recalculate it dynamically.

    # TODO we should set some flag on the resulting tensor from compute_weight,
    #  or maybe the owning module,
    #  such that we have a chance to tell RETURNN to use weight norm on the corresponding layer.

    return fn

  def remove(self, module: Module) -> None:
    weight = self.compute_weight(module)
    delattr(module, self.name)
    del module._parameters[self.name + '_g']
    del module._parameters[self.name + '_v']
    setattr(module, self.name, Parameter(weight.data))

  def __call__(self, module: Module, inputs: Any) -> None:
    setattr(module, self.name, self.compute_weight(module))


T_module = TypeVar('T_module', bound=Module)


def weight_norm(module: T_module, name: str = 'weight', dim: int = 0) -> T_module:
  WeightNorm.apply(module, name, dim)
  return module


def remove_weight_norm(module: T_module, name: str = 'weight') -> T_module:
  for k, hook in module._forward_pre_hooks.items():
    if isinstance(hook, WeightNorm) and hook.name == name:
      hook.remove(module)
      del module._forward_pre_hooks[k]
      return module
  raise ValueError(f"weight_norm of {name!r} not found in {module}")
