
import _setup_test_env  # noqa
import sys
import unittest
import typing
import numpy
from pytorch_to_returnn import torch
from pytorch_to_returnn.verify import verify_torch


def test_conv_transposed():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    # {'class': 'transposed_conv', 'from': 'layer2', 'activation': None, 'with_bias': True,
    #  'n_out': 192, 'filter_size': (10,), 'strides': (5,), 'remove_padding': (3,), 'output_padding': (1,)}
    model = torch.nn.ConvTranspose1d(
      in_channels=n_in,
      out_channels=n_out,
      kernel_size=10,
      stride=5,
      padding=3,
      output_padding=1)
    return model(inputs)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch(model_func, inputs=x)


def test_functional_conv():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7
  kernel_size = 3

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
      import torch.nn.functional as F
    else:
      torch = wrapped_import("torch")
      F = wrapped_import("torch.nn.functional")
    rnd = numpy.random.RandomState(42)
    weight = rnd.normal(0., 1., (n_out, n_in, kernel_size)).astype("float32")
    bias = rnd.normal(0., 1., (n_out,)).astype("float32")
    weight = torch.from_numpy(weight)
    bias = torch.from_numpy(bias)
    return F.conv1d(
      inputs,
      weight=weight, bias=bias,
      stride=2)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch(model_func, inputs=x)


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
