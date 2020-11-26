
import typing
from pytorch_to_returnn.import_wrapper import wrapped_import_torch_traced


def import_torch_traced():
  if typing.TYPE_CHECKING:
    import torch
    return torch

  return wrapped_import_torch_traced("torch")


def import_torch_returnn_wrapped():
  if typing.TYPE_CHECKING:
    from pytorch_to_returnn import torch
    return torch

  # TODO ...
  from pytorch_to_returnn import torch
  return torch


def def_model():
  torch = import_torch_returnn_wrapped()
  nn = torch.nn
  F = nn.functional

  # directly from here: https://github.com/pytorch/examples/blob/master/mnist/main.py
  class Net(torch.nn.Module):
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

  return Net()

