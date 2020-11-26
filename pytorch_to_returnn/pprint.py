
"""
Alternative to the original pprint module.
This one has different behavior for indentation, specifically for dicts.
Also the order of dict items are kept as-is
(which is fine for newer Python versions, which will be the insertion order).

Compare (via our ``pprint``)::

  {
    'melgan': {
      'class': 'subnetwork',
      'from': 'data',
      'subnetwork': {
        'l0': {'class': 'pad', 'mode': 'reflect', 'axes': 'spatial', 'padding': (3, 3), 'from': 'data'},
        'la1': {
          'class': 'conv',
          'from': 'l0',
          'activation': None,
          'with_bias': True,
          'n_out': 384,
          'filter_size': (7,),
          'padding': 'valid',
          'strides': (1,),
          'dilation_rate': (1,)
        },
        'lay2': {'class': 'eval', 'eval': 'tf.nn.leaky_relu(source(0), alpha=0.2)', 'from': 'la1'},
        'layer3_xxx': {
          'class': 'transposed_conv',
          'from': 'lay2',
          'activation': None,
          'with_bias': True,
          'n_out': 192,
          'filter_size': (10,),
          'strides': (5,),
          'padding': 'valid',
          'output_padding': (1,),
          'remove_padding': (3,)
        },
        'output': {'class': 'copy', 'from': 'layer3_xxx'}
      }
    },
    'output': {'class': 'copy', 'from': 'melgan'}
  }

Vs (via original ``pprint``)::

  {'melgan': {'class': 'subnetwork',
              'from': 'data',
              'subnetwork': {'l0': {'axes': 'spatial',
                                    'class': 'pad',
                                    'from': 'data',
                                    'mode': 'reflect',
                                    'padding': (3, 3)},
                             'la1': {'activation': None,
                                     'class': 'conv',
                                     'dilation_rate': (1,),
                                     'filter_size': (7,),
                                     'from': 'l0',
                                     'n_out': 384,
                                     'padding': 'valid',
                                     'strides': (1,),
                                     'with_bias': True},
                             'lay2': {'class': 'eval',
                                      'eval': 'tf.nn.leaky_relu(source(0), '
                                              'alpha=0.2)',
                                      'from': 'la1'},
                             'layer3_xxx': {'activation': None,
                                            'class': 'transposed_conv',
                                            'filter_size': (10,),
                                            'from': 'lay2',
                                            'n_out': 192,
                                            'output_padding': (1,),
                                            'padding': 'valid',
                                            'remove_padding': (3,),
                                            'strides': (5,),
                                            'with_bias': True},
                             'output': {'class': 'copy', 'from': 'layer3_xxx'}}},
   'output': {'class': 'copy', 'from': 'melgan'}}

This is a very simple implementation.
There are other similar alternatives:

* [Rich](https://github.com/willmcgugan/rich)
* [pprint++](https://github.com/wolever/pprintpp)

"""

from typing import Any
import sys
import numpy


def pprint(o: Any, *, file=sys.stdout,
           prefix="", postfix="",
           line_prefix="", line_postfix="\n") -> None:
  if "\n" in line_postfix and _type_simplicity_score(o) <= _type_simplicity_limit:
    prefix = f"{line_prefix}{prefix}"
    line_prefix = ""
    postfix = postfix + line_postfix
    line_postfix = ""

  def _sub_pprint(o: Any, prefix="", postfix="", inc_indent=True):
    multi_line = "\n" in line_postfix
    if not multi_line and postfix.endswith(","):
      postfix += " "
    pprint(
      o, file=file, prefix=prefix, postfix=postfix,
      line_prefix=(line_prefix + "  " * inc_indent) if multi_line else "",
      line_postfix=line_postfix)

  def _print(s: str, is_end: bool = False):
    nonlocal prefix  # no need for is_begin, just reset prefix
    file.write(line_prefix)
    file.write(prefix)
    file.write(s)
    if is_end:
      file.write(postfix)
    file.write(line_postfix)
    if "\n" in line_postfix:
      file.flush()
    prefix = ""

  def _print_list():
    for i, v in enumerate(o):
      _sub_pprint(v, postfix="," if i < len(o) - 1 else "")

  if isinstance(o, list):
    if len(o) == 0:
      _print("[]", is_end=True)
      return
    _print("[")
    _print_list()
    _print("]", is_end=True)
    return

  if isinstance(o, tuple):
    if len(o) == 0:
      _print("()", is_end=True)
      return
    if len(o) == 1:
      _sub_pprint(o[0], prefix=f"{prefix}(", postfix=f",){postfix}", inc_indent=False)
      return
    _print("(")
    _print_list()
    _print(")", is_end=True)
    return

  if isinstance(o, set):
    if len(o) == 0:
      _print("set()", is_end=True)
      return
    _print("{")
    _print_list()
    _print("}", is_end=True)
    return

  if isinstance(o, dict):
    if len(o) == 0:
      _print("{}", is_end=True)
      return
    _print("{")
    for i, (k, v) in enumerate(o.items()):
      _sub_pprint(v, prefix=f"{k!r}: ", postfix="," if i < len(o) - 1 else "")
    _print("}", is_end=True)
    return

  if isinstance(o, numpy.ndarray):
    _sub_pprint(
      o.tolist(),
      prefix=f"{prefix}numpy.array(",
      postfix=f", dtype=numpy.{o.dtype}){postfix}",
      inc_indent=False)
    return

  # fallback
  _print(repr(o), is_end=True)


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
  if isinstance(o, (tuple, list, set)):
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
