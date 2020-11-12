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
from pytorch_to_returnn import torch

class Model(torch.nn.Module):
 ...
```
