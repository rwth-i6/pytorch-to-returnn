
from typing import Any
from .. import log


class WrappedObject:
  def __init__(self, orig_obj: Any, name: str):
    self._wrapped__name = name
    self._wrapped__orig_obj = orig_obj

  def __repr__(self):
    if self._wrapped__orig_obj is self:
      return super(WrappedObject, self).__repr__()
    return "<WrappedObject %r type %s>" % (self._wrapped__name, type(self._wrapped__orig_obj))

  def __getattr__(self, item):
    if item == "_wrapped__orig_obj":
      # Special case. torch.Tensor functions would create new copies,
      # which would use our custom class, but then this is not initialized.
      return self
    if item == "_wrapped__name":
      return "<unknown>"
    if self._wrapped__orig_obj is self:  # special case
      raise AttributeError("No attrib %r" % item)
    res_ = getattr(self._wrapped__orig_obj, item)
    try:
      res = _wrap(res_, name="%s.%s" % (self._wrapped__name, item))
      if res is not res_:
        object.__setattr__(self, item, res)
      else:
        if log.Verbosity >= 10:
          log.unique_print("*** not wrapped: '%s.%s', type %r" % (self._wrapped__name, item, type(res)))
      if log.Verbosity >= 6:
        postfix = ""
        if isinstance(res, (WrappedObject, WrappedModule)) or getattr(res, "__module__", "").startswith(_ModPrefix):
          postfix = " -> %r" % res
        log.unique_print("*** indirect getattr '%s.%s'%s" % (self._wrapped__name, item, postfix))
    except AttributeError as exc:  # no exception expected. esp, we should **NOT** forward AttributeError
      raise RuntimeError(exc) from exc
    return res

  # setattr on this object directly, not on the wrapped one
  __setattr__ = object.__setattr__

  def __delattr__(self, item):
    if hasattr(self._wrapped__orig_obj, item):
      delattr(self._wrapped__orig_obj, item)
    if hasattr(self, item):
      super(WrappedObject, self).__delattr__(item)

  def __dir__(self):
    if self._wrapped__orig_obj is self:
      return []
    return dir(self._wrapped__orig_obj)

  def __bool__(self):
    if self._wrapped__orig_obj is self:
      return super(WrappedObject, self).__bool__(self)
    return bool(self._wrapped__orig_obj)
