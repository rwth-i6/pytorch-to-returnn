#!/usr/bin/env python3

"""
Copied many things from https://github.com/pytorch/examples/blob/master/mnist/main.py.
"""

import _setup_env  # noqa
import better_exchook
import typing
from pytorch_to_returnn.converter import verify_torch_and_convert_to_returnn
import torch
import torch.utils.data
from torchvision import datasets, transforms
import argparse
import os


def model_func(wrapped_import, inputs):
  if wrapped_import:
    model = wrapped_import("model")
  else:
    import model
  net = model.Net()
  net = net.eval()  # disable dropout
  return net(inputs)


my_dir = os.path.dirname(os.path.abspath(__file__))


def main():
  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                      help='input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                      help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs', type=int, default=14, metavar='N',
                      help='number of epochs to train (default: 14)')
  parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                      help='learning rate (default: 1.0)')
  parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                      help='Learning rate step gamma (default: 0.7)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--dry-run', action='store_true', default=False,
                      help='quickly check a single pass')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--save-model', action='store_true', default=False,
                      help='For Saving the current Model')
  args = parser.parse_args()
  use_cuda = not args.no_cuda and torch.cuda.is_available()

  torch.manual_seed(args.seed)

  device = torch.device("cuda" if use_cuda else "cpu")

  train_kwargs = {'batch_size': args.batch_size}
  test_kwargs = {'batch_size': args.test_batch_size}
  if use_cuda:
    cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
  ])
  dataset1 = datasets.MNIST(f'{my_dir}/../data', train=True, download=True, transform=transform)
  dataset2 = datasets.MNIST(f'{my_dir}/../data', train=False, transform=transform)
  train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
  test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

  data, target = next(iter(train_loader))
  data_np = data.numpy()

  verify_torch_and_convert_to_returnn(
    model_func, inputs=data_np, inputs_data_kwargs={"shape": (1, 28, 28)})


if __name__ == '__main__':
  better_exchook.install()
  main()
