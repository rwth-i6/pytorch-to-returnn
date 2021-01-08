#!/usr/bin/env python3


from typing import Optional, TextIO
import types
import torch
import torch.nn.functional
import pytorch_to_returnn.torch as torch_returnn
import pytorch_to_returnn.torch.nn.functional as torch_returnn_nn_functional


def type_name(obj):
  if isinstance(obj, (types.FunctionType, types.BuiltinFunctionType)):
    return "function"
  if isinstance(obj, type):
    if issubclass(obj, torch.nn.Module):
      return "torch.nn.Module"
    return "class"
  return type(obj).__name__


_IgnoreNameList = {
  # Stuff from typing.
  "TYPE_CHECKING", "Tuple", "Optional", "Union", "Any", "Sequence", "List",
  # Other imports:
  "torch",
  "os",
  "warnings",
}

_IgnoreFullnameList = {
  "torch.os",
  "torch.sys",
  "torch.torch",
  "torch.nn.functional.math",
}


def compare(prefix: str, orig_mod, our_mod):
  # Keep the order as-is. Should be like defined in the source?
  for name, value in vars(orig_mod).items():
    if name.startswith("_") or name in _IgnoreNameList:
      continue
    full_name = f"{prefix}.{name}"
    if full_name in _IgnoreFullnameList:
      continue

    report_available(full_name, value, available=hasattr(our_mod, name))


markdown_output_file = None  # type: Optional[TextIO]


def report_available(name: str, obj, available: bool):
  # Potential available True check marks: ✓✅✓✓☑
  # Potential available False check marks: ✗❎✗✘❌
  avail_s = "[x]" if available else "[ ]"
  print(avail_s, name, f"({type_name(obj)})")
  markdown_output_file.write(
    f"- {avail_s} `{name}` ({type_name(obj)})\n")


def main():
  global markdown_output_file
  markdown_output_file = open("Supported.md", "w")
  markdown_output_file.write("This file is auto-generated and should reflect the current state.\n\n")
  markdown_output_file.write("This is a list of supported PyTorch modules and functions.\n")
  markdown_output_file.write("See also [what is unsupported (on a higher level)](Unsupported.md).\n")
  markdown_output_file.write("\n")

  compare("torch.nn", torch.nn, torch_returnn.nn)
  compare("torch.nn.functional", torch.nn.functional, torch_returnn_nn_functional)
  compare("torch", torch, torch_returnn)

  markdown_output_file.close()


if __name__ == '__main__':
  main()
