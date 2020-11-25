
import _setup_test_env  # noqa
import sys
import unittest
import typing
import numpy
from pytorch_to_returnn.pprint import pprint, pformat, _type_simplicity_limit, _type_simplicity_score
from pprint import pprint as orig_pprint


def assert_equal(a_obj, b):
  a = pformat(a_obj)
  a = a.strip()
  b = b.strip()
  print("Repr:", repr(a_obj))
  print("Orig pprint:")
  orig_pprint(a_obj)
  print("New pprint:")
  print(a)
  if a != b:
    print("!=")
    print(b)
  assert a == b


class Obj:
  """
  Custom object to enforce to fallback to generic repr.
  """
  def __init__(self, value):
    self.value = value

  def __repr__(self):
    return f"{self.__class__.__name__}({self.value!r})"


def test_pprint_simple():
  assert_equal({}, "{}")
  assert_equal((), "()")
  assert_equal([], "[]")
  assert_equal(set(), "set()")


def test_pprint_simple_in_single_line():
  assert_equal({1}, "{1}")
  assert_equal({1, 2, 3}, "{1, 2, 3}")


def test_pprint_multi_line():
  assert_equal([Obj(1), 2, 3], """
[
  Obj(1),
  2,
  3
]""")
  assert_equal({"a": Obj(3)}, """
{
  'a': Obj(3)
}""")
  assert_equal({"a": Obj(3), "b": 4}, """
{
  'a': Obj(3),
  'b': 4
}""")
  net_dict_example = {
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
  assert_equal(net_dict_example, """
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
}""")


def test_pprint_numpy_ndarray():
  assert_equal(numpy.array([[1, 2], [3, 4]]), "numpy.array([[1, 2], [3, 4]], dtype=numpy.int64)")


def test_pprint_dict_numpy_ndarray():
  assert_equal(
    {"a": 42, "b": numpy.array([[1, 2, 3, 4]] * 10)}, """
{
  'a': 42,
  'b': numpy.array([
""" +
"""    [1, 2, 3, 4],
""" * 9 +
"""    [1, 2, 3, 4]
  ], dtype=numpy.int64)
}
""")


if __name__ == "__main__":
  if len(sys.argv) <= 1:
    for k, v in sorted(globals().items()):
      if k.startswith("test_"):
        print("-" * 40)
        print("Executing: %s" % k)
        try:
          v()
        except unittest.SkipTest as exc:
          print("SkipTest:", exc)
        print("-" * 40)
    print("Finished all tests.")
  else:
    assert len(sys.argv) >= 2
    for arg in sys.argv[1:]:
      print("Executing: %s" % arg)
      if arg in globals():
        globals()[arg]()  # assume function and execute
      else:
        eval(arg)  # assume Python code and execute
