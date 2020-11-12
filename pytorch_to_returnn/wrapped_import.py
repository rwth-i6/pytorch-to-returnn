"""
Utility to import a module with automatic Torch import wrapping, which replaces all::

  import torch

To::

  from pytorch_to_returnn import torch

In your user code, you would replace::

  import custom_torch_code

By::

  custom_torch_code = pytorch_to_returnn.wrapped_import.wrapped_import("custom_torch_code")

Both the wrapped and original module can be imported at the same time.
The wrapped module will internally get the full mod name ``pytorch_to_returnn._wrapped_mods.custom_torch_code``.
See :class:`_AstImportTransformer`.

"""

import torch
import typing
from typing import Union, Any
import sys
import importlib
import importlib.abc
import importlib.machinery
from . import log
from pytorch_to_returnn.import_wrapper.base_wrappers.object import WrappedObject
from pytorch_to_returnn.import_wrapper.base_wrappers.module import WrappedModule
from .import_wrapper.meta_path.loader import MetaPathLoader
from .import_wrapper.meta_path.finder import MetaPathFinder
from .import_wrapper.context import WrapCtx, make_torch_default_ctx


def _should_wrap_mod(mod_name: str) -> bool:
  if mod_name == "torch":
    return True
  if mod_name.startswith("torch."):
    return True
  return False


_ModPrefix = "%s._wrapped_mods." % __package__
_wrap_torch_ctx = make_torch_default_ctx(wrapped_mod_prefix=_ModPrefix)
_loader = MetaPathLoader(ctx=_wrap_torch_ctx)
_finder = MetaPathFinder(loader=_loader)


def wrapped_import(mod_name: str) -> Union[WrappedModule, Any]:  # rtype Any to make type checkers happy
  """
  :param str mod_name: full mod name, e.g. "custom_torch_code"
  :return: wrapped module
  """
  if _finder not in sys.meta_path:
    sys.meta_path.append(_finder)
  mod = importlib.import_module(_ModPrefix + mod_name)
  assert isinstance(mod, WrappedModule)
  return mod


if typing.TYPE_CHECKING:
  # This also currently does not work...
  # https://youtrack.jetbrains.com/issue/PY-45376
  wrapped_import = importlib.import_module
