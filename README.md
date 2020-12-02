Make [PyTorch](https://pytorch.org/) code
runnable within [RETURNN](https://github.com/rwth-i6/returnn).
This provides some wrappers (and maybe some magic) to do that.


# `torch` drop-in replacement for RETURNN

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

This would convert the model to a RETURNN model.
[Example constructed RETURNN net dict](https://gist.github.com/albertz/01264cfbd2dfd73a19c1e2ac40bdb16b),
created from
[this PyTorch code](https://github.com/albertz/import-parallel-wavegan/blob/main/pytorch_to_returnn.py).

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


# Import wrapper

We also support to transform external PyTorch code
on-the-fly
(without the need to rewrite the code;
it translates the code on AST level in the way above on-the-fly).
This is via our [generic Python import wrapper `pytorch_to_returnn.import_wrapper`](pytorch_to_returnn/import_wrapper).

Example for [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN):
```
from pytorch_to_returnn.import_wrapper import wrapped_import_torch_returnn
from pytorch_to_returnn.naming import Naming
from returnn.tf.util.data import Data

torch = wrapped_import_torch_returnn("torch")
wrapped_import_torch_returnn("parallel_wavegan")
pwg_models = wrapped_import_torch_returnn("parallel_wavegan.models")
pwg_layers = wrapped_import_torch_returnn("parallel_wavegan.layers")

with Naming.make_instance() as naming:
    inputs = torch.from_numpy(inputs)  # shape (Batch,Channel,Feature), e.g. (1,80,80)
    x = naming.register_input(
        inputs, Data("data", shape=(80, None), feature_dim_axis=1, time_dim_axis=2))
    assert isinstance(x, Data)

    # Initialize PWG
    pwg_config = yaml.load(open(args.pwg_config), Loader=yaml.Loader)
    generator = pwg_models.MelGANGenerator(**pwg_config['generator_params'])
    generator.load_state_dict(
        torch.load(args.pwg_checkpoint, map_location="cpu")["model"]["generator"])
    generator.remove_weight_norm()
    pwg_model = generator.eval()
    pwg_pqmf = pwg_layers.PQMF(pwg_config["generator_params"]["out_channels"])
    
    outputs = pwg_pqmf.synthesis(pwg_model(inputs))

    outputs = naming.register_output(outputs)
    y = outputs.returnn_data
    assert isinstance(y, Data)

```


# Model converter

For the process of converting a model from PyTorch to RETURNN,
including a PyTorch model checkpoint,
we provide some utilities to automate this,
and verify whether all outputs match.
This is in [`pytorch_to_returnn.converter`](pytorch_to_returnn/converter).

Example for [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN):
```
def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
        import torch
        from parallel_wavegan import models as pwg_models
        from parallel_wavegan import layers as pwg_layers

    else:
        torch = wrapped_import("torch")
        wrapped_import("parallel_wavegan")
        pwg_models = wrapped_import("parallel_wavegan.models")
        pwg_layers = wrapped_import("parallel_wavegan.layers")

    # Initialize PWG
    pwg_config = yaml.load(open(args.pwg_config), Loader=yaml.Loader)
    generator = pwg_models.MelGANGenerator(**pwg_config['generator_params'])
    generator.load_state_dict(
        torch.load(args.pwg_checkpoint, map_location="cpu")["model"]["generator"])
    generator.remove_weight_norm()
    pwg_model = generator.eval()
    pwg_pqmf = pwg_layers.PQMF(pwg_config["generator_params"]["out_channels"])

    return pwg_pqmf.synthesis(pwg_model(inputs))


feature_data = numpy.load(args.features)  # shape (Batch,Channel,Time) (1,80,80)

from pytorch_to_returnn.converter import verify_torch_and_convert_to_returnn
verify_torch_and_convert_to_returnn(model_func, inputs=feature_data)
```

This will automatically do the conversion,
i.e. create a RETURNN model,
including the [RETURNN net dict](https://gist.github.com/albertz/01264cfbd2dfd73a19c1e2ac40bdb16b)
and TF checkpoint file,
and do verification on several steps of all the outputs.


# Direct use in RETURNN

```
from pytorch_to_returnn import torch as torch_returnn

class MyTorchModel(torch_returnn.nn.Module):
  ...

my_torch_model = MyTorchModel() 

extern_data = {...}  # as usual

# RETURNN network dict
network = {
"prenet": my_torch_model.as_returnn_layer_dict(extern_data["data"]),

# Other RETURNN layers
...
}
```

Or:

```
from pytorch_to_returnn import torch as torch_returnn

class MyTorchModel(torch_returnn.nn.Module):
  ...

my_torch_model = MyTorchModel() 

extern_data = {...}  # as usual

# RETURNN network dict
network = my_torch_model.as_returnn_net_dict(extern_data["data"])
```
