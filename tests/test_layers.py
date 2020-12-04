
import _setup_test_env  # noqa
import sys
import unittest
import typing
import numpy
from pytorch_to_returnn import torch
from pytorch_to_returnn.converter import verify_torch_and_convert_to_returnn
from pytorch_to_returnn.pprint import pformat


def test_embedding():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    model = torch.nn.Embedding(n_in, n_out)
    return model(inputs)

  rnd = numpy.random.RandomState(42)
  x = rnd.randint(0, n_in, (n_time, n_batch), dtype="int64")
  verify_torch_and_convert_to_returnn(
    model_func,
    inputs=x, inputs_data_kwargs={"shape": (None,), "sparse": True, "dim": n_in, "batch_dim_axis": 1})


def test_conv():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    # {'class': 'transposed_conv', 'from': 'layer2', 'activation': None, 'with_bias': True,
    #  'n_out': 192, 'filter_size': (10,), 'strides': (5,), 'remove_padding': (3,), 'output_padding': (1,)}
    model = torch.nn.Conv1d(
      in_channels=n_in,
      out_channels=n_out,
      kernel_size=3,
      stride=2)
    return model(inputs)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_conv2d():
  n_in, n_out = 11, 13
  n_batch, n_time1, n_time2 = 3, 17, 19

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    # {'class': 'transposed_conv', 'from': 'layer2', 'activation': None, 'with_bias': True,
    #  'n_out': 192, 'filter_size': (10,), 'strides': (5,), 'remove_padding': (3,), 'output_padding': (1,)}
    model = torch.nn.Conv2d(
      in_channels=n_in,
      out_channels=n_out,
      kernel_size=(3, 5),
      stride=2)
    return model(inputs)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time1, n_time2)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


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
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


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
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_functional_conv_no_bias():
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
    weight = torch.from_numpy(weight)
    return F.conv1d(inputs, weight=weight, stride=2)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_functional_conv2d():
  n_in, n_out = 11, 13
  n_batch, n_time1, n_time2 = 3, 17, 19
  kernel_size = (3, 5)

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
      import torch.nn.functional as F
    else:
      torch = wrapped_import("torch")
      F = wrapped_import("torch.nn.functional")
    rnd = numpy.random.RandomState(42)
    weight = rnd.normal(0., 1., (n_out, n_in) + kernel_size).astype("float32")
    bias = rnd.normal(0., 1., (n_out,)).astype("float32")
    weight = torch.from_numpy(weight)
    bias = torch.from_numpy(bias)
    return F.conv2d(
      inputs,
      weight=weight, bias=bias,
      stride=2)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time1, n_time2)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_functional_conv_transposed():
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
    weight = rnd.normal(0., 1., (n_in, n_out, kernel_size)).astype("float32")
    weight = torch.from_numpy(weight)
    return F.conv_transpose1d(inputs, weight=weight, stride=2)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_unsqueeze():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    return inputs.unsqueeze(2)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_unsqueeze2():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    return inputs.unsqueeze(3)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_movedim():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    x = inputs  # (B,F,T)
    x = torch.nn.Conv1d(n_in, n_out, 3)(x)  # make it (B,T,F) in RETURNN. (B,F,T) in Torch.
    x = torch.movedim(x, 0, 1)  # stay (B,T,F) in RETURNN. (F,B,T) in Torch.
    return x

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_flatten_batch():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    x = inputs  # (B,F,T)
    x = torch.movedim(x, 1, 0)  # (F,B,T)
    x = torch.reshape(x, (x.shape[0], -1))  # (F,B*T)
    return x

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  converter = verify_torch_and_convert_to_returnn(model_func, inputs=x)
  assert converter.returnn_net_dict["Flatten"]["class"] == "flatten_batch", pformat(converter.returnn_net_dict)


def test_slice_1d():
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    x = inputs[0]  # (T)
    x = x[:3]
    return x

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_slice_2d():
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    x = inputs[:, 3:]
    return x

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_const_with_batch_and_gather():
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    batch_size = inputs.shape[0]
    x = inputs.new_zeros((2, batch_size, 13))  # const, shape (2,B,13). in RETURNN (B,2,13)
    x = x[0]  # shape (B,13). in RETURNN (B,13).
    x = x + inputs[:, :1]  # (B,13). just do this to have dep on inputs
    return x

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


@unittest.skip("not ready yet ... multiple outputs not supported")  # TODO...
def test_lstm_2l():
  n_batch, n_time = 3, 7
  n_in, n_out = 11, 13
  n_layers = 2

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    batch_size = inputs.shape[1]
    hidden = inputs.new_zeros((n_layers, batch_size, n_out))  # const, shape (2,B,13). in RETURNN (B,2,13)
    hidden = (hidden, hidden)
    lstm = torch.nn.LSTM(n_in, n_out, n_layers)
    out, hidden_ = lstm(inputs, hidden)
    return out

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_time, n_batch, n_in)).astype("float32")
  verify_torch_and_convert_to_returnn(
    model_func, inputs=x,
    inputs_data_kwargs={"shape": (None, n_in), "batch_dim_axis": 1})


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
