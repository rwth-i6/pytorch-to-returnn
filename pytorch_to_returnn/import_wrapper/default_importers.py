

from typing import Union, Any
from .base_wrappers.module import WrappedModule
from .context import make_torch_default_ctx, make_torch_demo_ctx
from .import_ import import_module


_ModPrefix = "%s._traced_torch." % __package__
_wrap_torch_ctx = make_torch_default_ctx(wrapped_mod_prefix=_ModPrefix)

_DemoModPrefix = "%s._torch_stub." % __package__
_wrap_torch_demo_ctx = make_torch_demo_ctx(wrapped_mod_prefix=_DemoModPrefix)


def wrapped_import(mod_name: str) -> Union[WrappedModule, Any]:  # rtype Any to make type checkers happy
  """
  :param str mod_name: full mod name, e.g. "custom_torch_code"
  :return: wrapped module
  """
  return import_module(mod_name, ctx=_wrap_torch_ctx)


def wrapped_import_demo(mod_name: str) -> Union[WrappedModule, Any]:  # rtype Any to make type checkers happy
  """
  :param str mod_name: full mod name, e.g. "custom_torch_code"
  :return: wrapped module
  """
  return import_module(mod_name, ctx=_wrap_torch_demo_ctx)
