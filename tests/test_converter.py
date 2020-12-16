
import _setup_test_env  # noqa
import sys
import unittest
import typing
import numpy
from pytorch_to_returnn import torch
from pytorch_to_returnn.converter import verify_torch_and_convert_to_returnn


def test_mnist():
  def model_func(wrapped_import, inputs):
    if wrapped_import:
      torch = wrapped_import("torch")
      nn = wrapped_import("torch.nn")
      F = wrapped_import("torch.nn.functional")
    else:
      import torch
      import torch.nn as nn
      import torch.nn.functional as F

    # directly from here: https://github.com/pytorch/examples/blob/master/mnist/main.py
    class Net(nn.Module):
      def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

      def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    net = Net()
    net = net.eval()  # disable dropout
    return net(inputs)

  rnd = numpy.random.RandomState(42)
  N, C, H, W = 64, 1, 28, 28
  x = rnd.normal(0., 1., (N, C, H, W)).astype("float32")
  verify_torch_and_convert_to_returnn(
    model_func, inputs=x, inputs_data_kwargs={"shape": (C, H, W)})


def test_weight_norm():
  n_batch, n_time = 3, 7
  n_in, n_out = 11, 13

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")

    class OuterModel(torch.nn.Module):
      def __init__(self):
        super(OuterModel, self).__init__()
        self.model = torch.nn.Conv1d(n_in, n_out, (1,))
        torch.nn.utils.weight_norm(self.model)

      def forward(self, x):
        return self.model(x)

    model = OuterModel()
    return model(inputs)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)
  # TODO actually also check that weight norm is used in RETURNN...


def test_custom_layer_norm():
  N, F, T = 64, 11, 28

  def model_func(wrapped_import, inputs):
    if wrapped_import:
      torch = wrapped_import("torch")
    else:
      import torch

    class LayerNorm(torch.nn.LayerNorm):
      def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

      def forward(self, x):
        if self.dim == -1:
          return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)

    class Net(torch.nn.Module):
      def __init__(self):
        super(Net, self).__init__()
        self.norm = LayerNorm(nout=F, dim=1)

      def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

    net = Net()
    return net(inputs)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (N, F, T)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_naming_problem():
  N, F, T = 3, 5, 11

  def model_func(wrapped_import, inputs):
    if wrapped_import:
      torch = wrapped_import("torch")
    else:
      import torch

    class Activation(torch.nn.Module):
      def forward(self, x):
        return torch.nn.GELU()(x)

    class ConvFeatureExtractionModel(torch.nn.Module):
      def __init__(self):
        super(ConvFeatureExtractionModel, self).__init__()
        self.conv_layers = torch.nn.ModuleList()
        in_d = F
        conv_layers = [(7, 3), (7, 3)]
        for i, (dim, k) in enumerate(conv_layers):
          self.conv_layers.append(
            torch.nn.Sequential(
              torch.nn.Conv1d(in_d, dim, k),
              Activation()))
          in_d = dim

      def forward(self, x):
        for conv in self.conv_layers:
          x = conv(x)
        return x

    class MainModel(torch.nn.Module):
      def __init__(self):
        super(MainModel, self).__init__()
        self.feature_extractor = ConvFeatureExtractionModel()

      def forward(self, x):
        return self.feature_extractor(x)

    model = MainModel()
    return model(inputs)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (N, F, T)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_naming_inner_func():
  n_batch, n_time = 3, 7
  n_in, n_out = 11, 13

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")

    class OuterModel(torch.nn.Module):
      def __init__(self):
        super(OuterModel, self).__init__()
        self.model = torch.nn.Conv1d(n_in, n_out, (1,))  # whatever, doesn't matter

      def some_func(self, x):  # intentionally this is not `forward`
        return self.model(x)

    model = OuterModel()
    out = model.some_func(inputs)
    return out

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_naming_inner_func_functional_with_buffer():
  n_batch, n_time = 3, 7
  n_in, n_out = 11, 13

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
      import torch.nn.functional as F
    else:
      torch = wrapped_import("torch")
      F = wrapped_import("torch.nn.functional")

    class OuterModel(torch.nn.Module):
      def __init__(self):
        super(OuterModel, self).__init__()
        h_synthesis = numpy.ones((n_in, 3))
        synthesis_filter = torch.from_numpy(h_synthesis).float().unsqueeze(0)
        self.register_buffer("synthesis_filter", synthesis_filter)

      def some_func(self, x):  # intentionally this is not `forward`
        return F.conv1d(x, self.synthesis_filter)

    model = OuterModel()
    out = model.some_func(inputs)
    return out

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


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
