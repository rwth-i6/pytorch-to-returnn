
from typing import Any, Iterable, Iterator, Mapping, Optional, overload, Tuple, TypeVar, Union, Dict
from collections import OrderedDict
import operator
from itertools import islice
from .module import Module
from ...tensor import Tensor


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
