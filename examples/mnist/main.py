#!/usr/bin/env python3

import better_exchook
import typing
from pytorch_to_returnn.import_wrapper import wrapped_import_torch_traced


def import_torch_traced():
  if typing.TYPE_CHECKING:
    import torch
    return torch

  return wrapped_import_torch_traced("torch")


def def_model():
  # import torch
  # import torch.nn as nn
  # import torch.nn.functional as F
  from pytorch_to_returnn import torch
  from pytorch_to_returnn.torch import nn
  from pytorch_to_returnn.torch.nn import functional as F

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

  return Net()


def main():
  def_model()


if __name__ == '__main__':
  better_exchook.install()
  main()
