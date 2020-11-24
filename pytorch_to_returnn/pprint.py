
"""
Alternative to the original pprint module.
This one has different behavior for indentation, specifically for dicts.
Also the order of dict items are kept as-is
(which is fine for newer Python versions, which will be the insertion order).
"""

from typing import Any
import sys
from collections import deque
import numpy


def pprint(o: Any, file=sys.stdout, line_prefix="", prefix="", postfix="") -> None:
  if _type_simplicity_score(o) <= _type_simplicity_limit:
    if isinstance(o, numpy.ndarray):
      print(f"{line_prefix}{prefix}numpy.array({o.tolist()!r}, dtype=numpy.{o.dtype}){postfix}", file=file)
      return
    print(f"{line_prefix}{prefix}{o!r}{postfix}", file=file)
    return

  def _print_list():
    for i, v in enumerate(o):
      pprint(v, file=file, line_prefix=line_prefix + "  ", postfix="," if i < len(o) - 1 else "")

  if isinstance(o, list):
    if len(o) == 0:
      print(f"{line_prefix}{prefix}[]{postfix}", file=file)
      return
    print(f"{line_prefix}{prefix}[", file=file)
    _print_list()
    print(f"{line_prefix}]{postfix}", file=file)
    return

  if isinstance(o, deque):
    print(f"{line_prefix}{prefix}deque([", file=file)
    _print_list()
    print(f"{line_prefix}]){postfix}", file=file)
    return

  if isinstance(o, tuple):
    if len(o) == 0:
      print(f"{line_prefix}{prefix}(){postfix}", file=file)
      return
    if len(o) == 1:
      pprint(o[0], file=file, line_prefix=line_prefix, prefix=f"{prefix}(", postfix=f",){postfix}")
      return
    print(f"{line_prefix}{prefix}(", file=file)
    _print_list()
    print(f"{line_prefix}){postfix}", file=file)
    return

  if isinstance(o, set):
    if len(o) == 0:
      print(f"{line_prefix}{prefix}set(){postfix}", file=file)
      return
    print(f"{line_prefix}{prefix}{'{'}", file=file)
    _print_list()
    print(f"{line_prefix}{'}'}{postfix}", file=file)
    return

  if isinstance(o, dict):
    if len(o) == 0:
      print(f"{line_prefix}{prefix}{'{}'}{postfix}", file=file)
      return
    print(f"{line_prefix}{prefix}{'{'}", file=file)
    for i, (k, v) in enumerate(o.items()):
      pprint(
        v, file=file, line_prefix=line_prefix + "  ",
        prefix=f"{k!r}: ",
        postfix="," if i < len(o) - 1 else "")
    print(f"{line_prefix}{'}'}{postfix}", file=file)
    return

  if isinstance(o, numpy.ndarray):
    pprint(
      o.tolist(), file=file, line_prefix=line_prefix,
      prefix=f"{prefix}numpy.array(",
      postfix=f", dtype=numpy.{o.dtype}){postfix}")
    return

  # fallback
  print(f"{line_prefix}{prefix}{o!r}{postfix}", file=file)


def pformat(o: Any) -> str:
  import io
  s = io.StringIO()
  pprint(o, file=s)
  return s.getvalue()


_type_simplicity_limit = 120.  # magic number


def _type_simplicity_score(o: Any, _offset=0.) -> float:
  """
  :param Any o:
  :param float _offset:
  :return: a score, which is a very rough estimate of len(repr(o)), calculated efficiently
  """
  _spacing = 2.
  if isinstance(o, bool):
    return 4. + _offset
  if isinstance(o, (int, numpy.integer)):
    if o == 0:
      return 1. + _offset
    return 1. + numpy.log10(abs(o)) + _offset
  if isinstance(o, str):
    return 2. + len(o) + _offset
  if isinstance(o, (float, complex, numpy.number)):
    return len(repr(o)) + _offset
  if isinstance(o, (tuple, list, set, deque)):
    for x in o:
      _offset = _type_simplicity_score(x, _offset=_offset + _spacing)
      if _offset > _type_simplicity_limit:
        break
    return _offset
  if isinstance(o, dict):
    for x in o.values():  # ignore keys...
      _offset = _type_simplicity_score(x, _offset=_offset + 10. + _spacing)  # +10 for key
      if _offset > _type_simplicity_limit:
        break
    return _offset
  if isinstance(o, numpy.ndarray):
    _offset += 10.  # prefix/postfix
    if o.size * 2. + _offset > _type_simplicity_limit:  # too big already?
      return o.size * 2. + _offset
    if str(o.dtype).startswith("int"):
      a = _type_simplicity_score(numpy.max(numpy.abs(o))) + _spacing
      return o.size * a + _offset
    a = max([_type_simplicity_score(x) for x in o.flatten()]) + _spacing
    return o.size * a + _offset
  # Unknown object. Fallback > _type_simplicity_limit.
  return _type_simplicity_limit + 1. + _offset
