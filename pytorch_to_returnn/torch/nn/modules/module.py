

from collections import OrderedDict
from typing import Optional, Callable, TypeVar, Iterator, Tuple
from ..parameter import Parameter
from ...tensor import Tensor
from ...utils.hooks import RemovableHandle


# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.
T = TypeVar('T', bound='Module')


class Module:
  """
  Base class.
  """
  def __init__(self):
    self._parameters = OrderedDict()  # type: OrderedDict[str, Parameter]
    self._modules = OrderedDict()  # type: OrderedDict[str, Optional[Module]]
    self._buffers = OrderedDict()  # type: OrderedDict[str, Tensor]
    self._non_persistent_buffers_set = set()
    self._forward_pre_hooks = OrderedDict()

  def __getattr__(self, item):
    if item in {"_parameters", "_modules", "_buffers"}:
      raise AttributeError(f"Module.__init__ not yet called, attrib {item!r} not present")
    if item in self._parameters:
      return self._parameters[item]
    if item in self._buffers:
      return self._buffers[item]
    if item in self._modules:
      return self._modules[item]
    raise AttributeError(f"No attrib {item!r}")

  def __setattr__(self, key, value):
    if isinstance(value, Parameter):
      self._parameters[key] = value
      return
    if isinstance(value, Tensor):
      self._buffers[key] = value
      return
    if isinstance(value, Module):
      self._modules[key] = value
      return
    return super(Module, self).__setattr__(key, value)

  def __delattr__(self, item):
    if item in self._parameters:
      del self._parameters[item]
      return
    if item in self._buffers:
      del self._buffers[item]
      return
    if item in self._modules:
      del self._modules[item]
      return
    super(Module, self).__delattr__(item)

  def __dir__(self):
    module_attrs = dir(self.__class__)
    attrs = list(self.__dict__.keys())
    parameters = list(self._parameters.keys())
    modules = list(self._modules.keys())
    buffers = list(self._buffers.keys())
    keys = module_attrs + attrs + parameters + modules + buffers
    keys = [key for key in keys if not key[0].isdigit()]  # Eliminate attrs that are not legal Python variable names
    return sorted(keys)

  def children(self) -> Iterator['Module']:
    for name, module in self.named_children():
      yield module

  def named_children(self) -> Iterator[Tuple[str, 'Module']]:
    memo = set()
    for name, module in self._modules.items():
      if module is not None and module not in memo:
        memo.add(module)
        yield name, module

  def add_module(self, name: str, module: Optional['Module']) -> None:
    if not isinstance(module, Module) and module is not None:
      raise TypeError(f"{type(module)} is not a Module subclass")
    elif not isinstance(name, str):
      raise TypeError(f"module name should be a string. Got {type(name)}")
    elif hasattr(self, name) and name not in self._modules:
      raise KeyError(f"attribute {name!r} already exists")
    elif '.' in name:
      raise KeyError("module name can't contain \".\"")
    elif name == '':
      raise KeyError("module name can't be empty string \"\"")
    self._modules[name] = module

  def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
    if '_parameters' not in self.__dict__:
      raise AttributeError("cannot assign parameter before Module.__init__() call")
    elif not isinstance(name, str):
      raise TypeError(f"parameter name should be a string. Got {type(name)}")
    elif '.' in name:
      raise KeyError("parameter name can't contain \".\"")
    elif name == '':
      raise KeyError("parameter name can't be empty string \"\"")
    elif hasattr(self, name) and name not in self._parameters:
      raise KeyError("attribute '{}' already exists".format(name))
    self._parameters[name] = param

  def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
    if '_buffers' not in self.__dict__:
      raise AttributeError("cannot assign buffer before Module.__init__() call")
    elif not isinstance(name, str):
      raise TypeError(f"buffer name should be a string. Got {type(name)}")
    elif '.' in name:
      raise KeyError("buffer name can't contain \".\"")
    elif name == '':
      raise KeyError("buffer name can't be empty string \"\"")
    elif hasattr(self, name) and name not in self._buffers:
      raise KeyError("attribute '{}' already exists".format(name))
    elif tensor is not None and not isinstance(tensor, Tensor):
      raise TypeError(f"cannot assign {type(tensor)} object to buffer {name!r}")
    else:
      self._buffers[name] = tensor
      if persistent:
        self._non_persistent_buffers_set.discard(name)
      else:
        self._non_persistent_buffers_set.add(name)

  def apply(self: T, fn: Callable[['Module'], None]) -> T:
    for module in self.children():
      module.apply(fn)
    fn(self)
    return self

  def register_forward_pre_hook(self, hook: Callable[..., None]) -> RemovableHandle:
    handle = RemovableHandle(self._forward_pre_hooks)
    self._forward_pre_hooks[handle.id] = hook
    return handle

  def load_state_dict(self, *args, **kwargs):
    pass  # ignore

  def eval(self):
    return self  # ignore

  def train(self, arg):
    return self  # ignore

  def to(self, *args):
    return self  # ignore

  def forward(self, input: Tensor):
    raise NotImplementedError

  def __call__(self, *input, **kwargs):
    for hook in self._forward_pre_hooks.values():
      result = hook(self, input)
      if result is not None:
        if not isinstance(result, tuple):
          result = (result,)
        input = result
    return self.forward(*input, **kwargs)
