
from typing import Any, Iterable, Iterator, Mapping, Optional, overload, Tuple, TypeVar, Union, Dict
from collections import OrderedDict
from collections import abc as container_abcs
import operator
from itertools import islice
from .module import Module
from ...tensor import Tensor


T = TypeVar('T')


class Sequential(Module):
  @overload
  def __init__(self, *args: Module) -> None:
    ...

  @overload
  def __init__(self, arg: 'OrderedDict[str, Module]') -> None:
    ...

  def __init__(self, *args: Any):
    super(Sequential, self).__init__()
    if len(args) == 1 and isinstance(args[0], OrderedDict):
      for key, module in args[0].items():
        self.add_module(key, module)
    else:
      for idx, module in enumerate(args):
        self.add_module(str(idx), module)

  def _get_item_by_idx(self, iterator, idx):
    """Get the idx-th item of the iterator"""
    size = len(self)
    idx = operator.index(idx)
    if not -size <= idx < size:
      raise IndexError('index {} is out of range'.format(idx))
    idx %= size
    return next(islice(iterator, idx, None))

  def __getitem__(self, idx: Union[slice, int]):
    if isinstance(idx, slice):
      return self.__class__(OrderedDict(list(self._modules.items())[idx]))
    else:
      return self._get_item_by_idx(self._modules.values(), idx)

  def __setitem__(self, idx: int, module: Module) -> None:
    key = self._get_item_by_idx(self._modules.keys(), idx)
    return setattr(self, key, module)

  def __delitem__(self, idx: Union[slice, int]) -> None:
    if isinstance(idx, slice):
      for key in list(self._modules.keys())[idx]:
        delattr(self, key)
    else:
      key = self._get_item_by_idx(self._modules.keys(), idx)
      delattr(self, key)

  def __len__(self) -> int:
    return len(self._modules)

  def __iter__(self) -> Iterator[Module]:
    return iter(self._modules.values())

  def forward(self, input):
    for module in self:
      res = module(input)
      assert isinstance(res, Tensor)
      input = res
    return input


class ModuleList(Module):
  def __init__(self, modules: Optional[Iterable[Module]] = None) -> None:
    super(ModuleList, self).__init__()
    if modules is not None:
      self += modules

  def _get_abs_string_index(self, idx):
    """Get the absolute index for the list of modules"""
    idx = operator.index(idx)
    if not (-len(self) <= idx < len(self)):
      raise IndexError('index {} is out of range'.format(idx))
    if idx < 0:
      idx += len(self)
    return str(idx)

  def __getitem__(self, idx: int) -> Module:
    if isinstance(idx, slice):
      return self.__class__(list(self._modules.values())[idx])
    else:
      return self._modules[self._get_abs_string_index(idx)]

  def __setitem__(self, idx: int, module: Module) -> None:
    idx = self._get_abs_string_index(idx)
    return setattr(self, str(idx), module)

  def __delitem__(self, idx: Union[int, slice]) -> None:
    if isinstance(idx, slice):
      for k in range(len(self._modules))[idx]:
        delattr(self, str(k))
    else:
      delattr(self, self._get_abs_string_index(idx))
    # To preserve numbering, self._modules is being reconstructed with modules after deletion
    str_indices = [str(i) for i in range(len(self._modules))]
    self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

  def __len__(self) -> int:
    return len(self._modules)

  def __iter__(self) -> Iterator[Module]:
    return iter(self._modules.values())

  def __iadd__(self: T, modules: Iterable[Module]) -> T:
    return self.extend(modules)

  def __dir__(self):
    keys = super(ModuleList, self).__dir__()
    keys = [key for key in keys if not key.isdigit()]
    return keys

  def insert(self, index: int, module: Module) -> None:
    """Insert a given module before a given index in the list."""
    for i in range(len(self._modules), index, -1):
      self._modules[str(i)] = self._modules[str(i - 1)]
    self._modules[str(index)] = module

  def append(self: T, module: Module) -> T:
    """Appends a given module to the end of the list."""
    self.add_module(str(len(self)), module)
    return self

  def extend(self: T, modules: Iterable[Module]) -> T:
    """Appends modules from a Python iterable to the end of the list."""
    if not isinstance(modules, container_abcs.Iterable):
      raise TypeError("ModuleList.extend should be called with an "
                      "iterable, but got " + type(modules).__name__)
    offset = len(self)
    for i, module in enumerate(modules):
      self.add_module(str(offset + i), module)
    return self

  def forward(self):
    raise NotImplementedError()


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
