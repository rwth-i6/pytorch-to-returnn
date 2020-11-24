
import _setup_test_env  # noqa
import sys
import unittest
import typing
import numpy
from pytorch_to_returnn.pprint import pprint, pformat, _type_simplicity_limit, _type_simplicity_score


def assert_equal(a, b):
  a = a.strip()
  b = b.strip()
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
  assert_equal(pformat({}), "{}")
  assert_equal(pformat(()), "()")
  assert_equal(pformat([]), "[]")
  assert_equal(pformat(set()), "set()")


def test_pprint_simple_in_single_line():
  assert_equal(pformat({1}), "{1}")
  assert_equal(pformat({1, 2, 3}), "{1, 2, 3}")


def test_pprint_multi_line():
  assert_equal(pformat([Obj(1), 2, 3]), """
[
  Obj(1),
  2,
  3
]""")
  assert_equal(pformat({"a": Obj(3)}), """
{
  'a': Obj(3)
}""")
  assert_equal(pformat({"a": Obj(3), "b": 4}), """
{
  'a': Obj(3),
  'b': 4
}""")


def test_pprint_numpy_ndarray():
  assert_equal(pformat(numpy.array([[1, 2], [3, 4]])), "numpy.array([[1, 2], [3, 4]], dtype=numpy.int64)")


def test_pprint_dict_numpy_ndarray():
  assert_equal(
    pformat({"a": 42, "b": numpy.array([[1, 2, 3, 4]] * 10)}), """
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
