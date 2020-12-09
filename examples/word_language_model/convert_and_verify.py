#!/usr/bin/env python3

import _setup_env  # noqa
import argparse
import time
import math
import os
import torch
import better_exchook

import data


my_dir = os.path.dirname(os.path.abspath(__file__))


def main():

  parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
  parser.add_argument('--data', type=str, default=f'{my_dir}/data/wikitext-2',
                      help='location of the data corpus')
  parser.add_argument('--model', type=str, default='LSTM',
                      help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
  parser.add_argument('--emsize', type=int, default=200,
                      help='size of word embeddings')
  parser.add_argument('--nhid', type=int, default=200,
                      help='number of hidden units per layer')
  parser.add_argument('--nlayers', type=int, default=2,
                      help='number of layers')
  parser.add_argument('--lr', type=float, default=20,
                      help='initial learning rate')
  parser.add_argument('--clip', type=float, default=0.25,
                      help='gradient clipping')
  parser.add_argument('--epochs', type=int, default=40,
                      help='upper epoch limit')
  parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                      help='batch size')
  parser.add_argument('--bptt', type=int, default=35,
                      help='sequence length')
  parser.add_argument('--dropout', type=float, default=0.2,
                      help='dropout applied to layers (0 = no dropout)')
  parser.add_argument('--tied', action='store_true',
                      help='tie the word embedding and softmax weights')
  parser.add_argument('--seed', type=int, default=1111,
                      help='random seed')
  parser.add_argument('--cuda', action='store_true',
                      help='use CUDA')
  parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                      help='report interval')
  parser.add_argument('--save', type=str, default='model.pt',
                      help='path to save the final model')
  parser.add_argument('--onnx-export', type=str, default='',
                      help='path to export the final model in onnx format')

  parser.add_argument('--nhead', type=int, default=2,
                      help='the number of heads in the encoder/decoder of the transformer model')
  parser.add_argument('--dry-run', action='store_true',
                      help='verify the code and the model')

  args = parser.parse_args()

  # Set the random seed manually for reproducibility.
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
      if not args.cuda:
          print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  device = torch.device("cuda" if args.cuda else "cpu")

  ###############################################################################
  # Load data
  ###############################################################################

  corpus = data.Corpus(args.data)

  # Starting from sequential data, batchify arranges the dataset into columns.
  # For instance, with the alphabet as the sequence and batch size 4, we'd get
  # ┌ a g m s ┐
  # │ b h n t │
  # │ c i o u │
  # │ d j p v │
  # │ e k q w │
  # └ f l r x ┘.
  # These columns are treated as independent by the model, which means that the
  # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
  # batch processing.

  def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

  eval_batch_size = 10
  train_data = batchify(corpus.train, args.batch_size)
  val_data = batchify(corpus.valid, eval_batch_size)
  test_data = batchify(corpus.test, eval_batch_size)

  def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i: i +seq_len]
    target = source[ i +1: i + 1 +seq_len].view(-1)
    return data, target

  ntokens = len(corpus.dictionary)

  def model_func(wrapped_import, inputs):
    ###############################################################################
    # Build the model
    ###############################################################################
    if wrapped_import:
      nn = wrapped_import("torch.nn")
      model = wrapped_import("model")
    else:
      from torch import nn
      import model

    if args.model == 'Transformer':
      net = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout)
    else:
      net = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)

    net.eval()  # for verification, need no random elements (e.g. dropout)
    # criterion = nn.NLLLoss()

    if args.model != 'Transformer':
      hidden = net.init_hidden(bsz=inputs.shape[1])
    else:
      hidden = None
    with torch.no_grad():
      if args.model == 'Transformer':
        output = net(inputs)
        output = output.view(-1, ntokens)
      else:
        output, hidden = net(inputs, hidden)

      return output

  data_, targets = get_batch(train_data, i=0)

  from pytorch_to_returnn.converter import verify_torch_and_convert_to_returnn
  verify_torch_and_convert_to_returnn(
    model_func,
    inputs=data_.detach().cpu().numpy(),
    inputs_data_kwargs={"shape": (None,), "sparse": True, "dim": ntokens, "batch_dim_axis": 1})


if __name__ == '__main__':
  better_exchook.install()
  main()
