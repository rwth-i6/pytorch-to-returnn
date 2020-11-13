
import functools
import inspect
from typing import TypeVar, Callable, Any, cast


FuncType = Callable[..., Any]
F = TypeVar('F', bound=FuncType)


class _DecoratorContextManager:
  """Allow a context manager to be used as a decorator"""

  def __call__(self, func: F) -> F:
    if inspect.isgeneratorfunction(func):
      return self._wrap_generator(func)

    @functools.wraps(func)
    def decorate_context(*args, **kwargs):
      with self.__class__():
        return func(*args, **kwargs)

    return cast(F, decorate_context)

  def _wrap_generator(self, func):
    """Wrap each generator invocation with the context manager"""

    @functools.wraps(func)
    def generator_context(*args, **kwargs):
      gen = func(*args, **kwargs)
      while True:
        try:
          with self.__class__():
            x = next(gen)
          yield x
        except StopIteration:
          break

    return generator_context

  def __enter__(self) -> None:
    raise NotImplementedError

  def __exit__(self, exc_type, exc_value, traceback):
    raise NotImplementedError


class no_grad(_DecoratorContextManager):
  def __enter__(self):
    pass

  def __exit__(self, exc_type, exc_value, traceback):
    pass
