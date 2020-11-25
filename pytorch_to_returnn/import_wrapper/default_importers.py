

from typing import Union, Any
from .base_wrappers.module import WrappedModule
from .context import make_torch_traced_ctx, make_torch_returnn_ctx
from .import_ import import_module


_TorchTracedModPrefix = "%s._torch_traced." % __package__
_wrap_torch_traced_ctx = make_torch_traced_ctx(wrapped_mod_prefix=_TorchTracedModPrefix)

_TorchReturnnModPrefix = "%s._torch_returnn." % __package__
_wrap_torch_returnn_ctx = make_torch_returnn_ctx(wrapped_mod_prefix=_TorchReturnnModPrefix)


def wrapped_import_torch_traced(mod_name: str) -> Union[WrappedModule, Any]:  # rtype Any to make type checkers happy
  """
  :param str mod_name: full mod name, e.g. "custom_torch_code"
  :return: wrapped module
  """
  return import_module(mod_name, ctx=_wrap_torch_traced_ctx)


def wrapped_import_torch_returnn(mod_name: str) -> Union[WrappedModule, Any]:  # rtype Any to make type checkers happy
  """
  :param str mod_name: full mod name, e.g. "custom_torch_code"
  :return: wrapped module
  """
  return import_module(mod_name, ctx=_wrap_torch_returnn_ctx)
