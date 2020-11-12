
import types
from .object import WrappedObject
from ..context import WrapCtx


__all__ = ["WrappedModule", "WrappedSourceModule", "WrappedIndirectModule"]


class WrappedModule(types.ModuleType):
  """
  Base class for :class:`WrappedSourceModule` or :class:`WrappedIndirectModule`.
  """
  def __init__(self, *, name: str, orig_mod: types.ModuleType, ctx: WrapCtx):
    super(WrappedModule, self).__init__(name=name)
    self._wrapped__orig_mod = orig_mod
    self._wrapped__ctx = ctx
    assert not isinstance(orig_mod, WrappedModule)

  def __repr__(self):
    return "<%s %s>" % (self.__class__.__name__, self.__name__)


class WrappedSourceModule(WrappedModule):
  """
  Represents a module where the AST was transformed via :class:`AstImportTransformer`.
  """
  def __init__(self, *, source: str, **kwargs):
    super(WrappedSourceModule, self).__init__(**kwargs)
    self._wrapped__source = source


class WrappedIndirectModule(WrappedModule, WrappedObject):
  """
  This can not be transformed on source level,
  so we wrap all attrib access on-the-fly.
  """
  def __init__(self, **kwargs):
    WrappedModule.__init__(self, **kwargs)
    WrappedObject.__init__(self, orig_obj=self._wrapped__orig_mod, name=self.__name__, ctx=self._wrapped__ctx)
    if getattr(self._wrapped__orig_mod, "__all__", None) is not None:
      # noinspection PyUnresolvedReferences
      self.__all__ = self._wrapped__orig_mod.__all__
    else:
      names = sorted(vars(self._wrapped__orig_mod).keys())
      self.__all__ = [name for name in names if name[:1] != "_"]

  def __getattr__(self, item):
    assert item not in {"_wrapped__orig_mod", "__name__"}
    if item == "__path__":  # some special logic, to avoid loading source directly
      if getattr(self._wrapped__orig_mod, "__path__", None) is not None:
        return []
      return None
    return WrappedObject.__getattr__(self, item)

  # setattr not needed
  # but if so, standard keys:
  # {"_wrapped__orig_mod", "__all__", "__loader__", "__package__", "__spec__", "__name__", "__path__"}
