#!/usr/bin/env python3

import _setup_env  # noqa
import numpy
import better_exchook
from pytorch_to_returnn.converter import verify_torch_and_convert_to_returnn


Batch = 3
Time = 23
Dim = 19


def model_func(wrapped_import, inputs):
  if wrapped_import:
    conformer = wrapped_import("conformer")
  else:
    import conformer

  # https://github.com/lucidrains/conformer
  layer = conformer.ConformerConvModule(
    dim=Dim,
    causal=False,  # auto-regressive or not - 1d conv will be made causal with padding if so
    expansion_factor=2,  # what multiple of the dimension to expand for the depthwise convolution
    kernel_size=31,  # kernel size, 17 - 31 was said to be optimal
    dropout=0.  # dropout at the very end
  )
  layer.eval()  # disable dropout for the test

  x = inputs
  x = layer(x) + x
  return x


def main():
  inputs = numpy.random.randn(Batch, Time, Dim).astype("float32")
  verify_torch_and_convert_to_returnn(
    model_func, inputs=inputs, inputs_data_kwargs={
      "shape": (None, Dim), "time_dim_axis": 1, "feature_dim_axis": 2})


if __name__ == '__main__':
  better_exchook.install()
  main()
