


class _WrappedClassBase:
  pass


def make_wrapped_class(cls: type, name: str):
  is_torch_module = issubclass(cls, torch.nn.Module)

  # TODO use cls as base?
  class WrappedClass(WrappedObject, _WrappedClassBase):
    def __init__(self, *args, **kwargs):
      if log.Verbosity >= 4:
        log.unique_print("*** WrappedClass %s(...)" % (name,))
      if isinstance(kwargs.get("orig_obj"), cls):
        assert not args
        WrappedObject.__init__(self, **kwargs)
      else:
        obj = cls(*unwrap(args), **unwrap(kwargs))
        WrappedObject.__init__(self, orig_obj=obj, name="%s(...)" % name)

    def __getattribute__(self, item):
      #if item == "__dict__":
        # Special case. Directly get it from wrapped object.
      #  return self._wrapped__orig_obj.__dict__
      if item in {"_wrapped__orig_obj", "_wrapped__name", "__class__"}:  # extra checks, and fast path
        return object.__getattribute__(self, item)
      if is_torch_module and False:
        # Some fast path, and avoid logging.
        if item in _TorchModDirectAttribs:
          return getattr(self._wrapped__orig_obj, item)
      if self._wrapped__orig_obj is self:  # special case
        res = object.__getattribute__(self, item)
        res = wrap(res, name="%s.%s" % (self._wrapped__name, item))
        return res
      return object.__getattribute__(self, item)

    def __setattr__(self, key, value):
      if key in {"_wrapped__orig_obj", "_wrapped__name"}:
        return object.__setattr__(self, key, value)
      if self is self._wrapped__orig_obj:  # special case
        return object.__setattr__(self, key, value)
      return setattr(self._wrapped__orig_obj, key, value)

    if is_torch_module:
      def __call__(self, *args, **kwargs):
        if log.Verbosity >= 3:
          log.unique_print("*** module call %s(...)(...)" % (name,))
        return self._wrapped__orig_obj(*args, **kwargs)

  WrappedClass.__name__ = cls.__name__
  WrappedClass.__qualname__ = cls.__qualname__
  if cls.__module__:
    if _should_wrap_mod(cls.__module__):
      WrappedClass.__module__ = _ModPrefix + cls.__module__
    else:
      WrappedClass.__module__ = cls.__module__
  try:
    WrappedClass.__bases__ += (cls,)
  except TypeError:  # e.g. object layout differs or so
    pass  # just ignore
  return WrappedClass
