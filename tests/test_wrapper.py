
import _setup_test_env  # noqa
import sys
import unittest
import typing
import numpy
import torch
from pytorch_to_returnn import torch as torch_returnn
from pytorch_to_returnn.import_wrapper import wrapped_import_torch_traced


def assert_equal(msg, a, b):
  print(f"check {msg} {a!r} == {b!r}")
  assert a == b


def test_torch_traced_wrapped_tensor_new():
  torch_traced = wrapped_import_torch_traced("torch")
  x = torch.Tensor()
  x_ = torch_traced.Tensor()
  assert_equal("new_zeros(3).shape:", x.new_zeros(3).shape, x_.new_zeros(3).shape)
  assert_equal("new_zeros([3]).shape:", x.new_zeros([3]).shape, x_.new_zeros([3]).shape)
  assert_equal("new(3).shape:", x.new(3).shape, x_.new(3).shape)


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
