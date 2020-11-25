Make PyTorch code runnable within RETURNN.
This provides some wrappers (and maybe some magic) to do that.

The idea:
```
import torch

class Model(torch.nn.Module):
 ...
```
Would become:
```
from pytorch_to_returnn import torch as torch_returnn

class Model(torch_returnn.nn.Module):
 ...
```

Somewhat related is also the `torch.fx` module.

See the [documentation of the `pytorch_to_returnn.torch` package](pytorch_to_returnn/torch)
for details about how this works,
and what can be done with it.
Obviously, this is incomplete.
For some status of what is not supported currently,
see [the unsupported document](Unsupported.md).
Otherwise, when you hit some `Module`
or `functional` function, or Tensor function
which is not implemented,
it just means that no-one has implemented it yet.
 
We also support to transform external PyTorch code
on-the-fly
(without the need to rewrite the code;
it translates the code on AST level in the way above on-the-fly).
This is via our [`pytorch_to_returnn.import_wrapper`](pytorch_to_returnn/import_wrapper).

For the process of converting a model from PyTorch to RETURNN,
including a PyTorch model checkpoint,
we provide some utilities to automate this,
and verify whether all outputs match.
This is in [`pytorch_to_returnn.converter`](pytorch_to_returnn/converter).
