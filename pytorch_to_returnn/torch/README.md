This is supposed to be used as a drop-in replacement
for the PyTorch `torch` module.
Specifically, given code like this:
```
import torch

class Model(torch.nn.Module):
 ...
```
You can use this package instead:
```
from pytorch_to_returnn import torch as torch_returnn

class Model(torch_returnn.nn.Module):
 ...
```

This is obviously incomplete.
However, it tries to work for simple model definitions.


# What does it do

This will map the model definition to an equivalent RETURNN model.
It can either create RETURNN layers (`LayerBase` objects)
and a RETURNN network (`TFNetwork` object) on-the-fly,
or just create the RETURNN layer definition.
(*Only the on-the-fly creation currently works.
  However, the full RETURNN net dict can be dumped later.*)


# Use cases

* Convert a model written with PyTorch code
  into an equivalent RETURNN model.
  
  - Live `TFNetwork` instance.
    This can also import PyTorch parameters.
    This is useful when you use [RETURNN as a framework](https://returnn.readthedocs.io/en/latest/getting_started/framework.html)
    and might be useful for some custom scripts.
  - RETURNN net dict for your RETURNN config.
    You would copy & paste this into your RETURNN config,
    either directly as the `network`,
    or as a subnetwork.
    [Example constructed RETURNN net dict](https://gist.github.com/albertz/01264cfbd2dfd73a19c1e2ac40bdb16b),
    created from
    [this PyTorch code](https://github.com/albertz/import-parallel-wavegan/blob/main/pytorch_to_returnn.py).

* Convert a PyTorch parameters / model checkpoint
  into an equivalent RETURNN/TF model checkpoint.

  - If you directly use it as a model in RETURNN,
    you can directly load the converted TF checkpoint.
  - If you use it as a subnetwork in RETURNN,
    you can use RETURNN `preload_from_files` with a prefix.

* Directly embed the PyTorch code into your RETURNN config,
  and then write sth like:
  
  ```
  from pytorch_to_returnn import torch as torch_returnn
  
  class MyTorchModel(torch_returnn.nn.Module):
    ...
  
  my_torch_model = MyTorchModel() 
  
  # RETURNN network dict
  network = {
    "prenet":
      my_torch_model.as_returnn_layer_dict(extern_data["data"]),
    
    # Other RETURNN layers
    ...
  }
  ```


Note:
If you want to import other external PyTorch code,
you cannot simply import it as-is,
because that would not use `pytorch_to_returnn.torch`.
However, we provide an [import wrapper](../import_wrapper),
which would transform the code on-the-fly.


# How does it work

Most straight-forward are the `Module` classes, like `Conv1d`.
But also the `Module` base class is implemented,
such that this supports any custom `Module` with a `forward` function.
This also implements the `functional` API and basic `Tensor` functions,
where all operations wrap to specific Torch `Module`s.

## `Module`

Also see our [`Module` base class](nn/modules/module.py). 

### Endpoints

Directly wrapped modules like `Conv1d`
would not have the `forward` function,
but instead provide `create_returnn_layer_dict`.

Note that a `Module` can be called multiple times,
and would reuse the same internal parameters.
A RETURNN layer does not support this concept.
However, a RETURNN layer can reuse parameters from another layer.
Thus, if a `Module` is called a second time,
a new RETURNN layer will be created,
which shares the parameters with the previous layer.

### User modules, or non-endpoints

E.g. `Sequential` does provide a `forward` function
(and no `create_returnn_layer_dict`)
and this will be used then.

The same is true for all user modules,
in custom PyTorch code,
which is expected to work as-is.

Such a `Module` with a `forward`
will automatically be wrapped as a RETURNN subnetwork
(`SubnetworkLayer`).

`Parameter`s used in a non-endpoint layer
would wrap to `VariableLayer`,
and other constants (e.g. created during `__init__`)
would wrap to `ConstantLayer`. 

## `Tensor`

Everything would use our own `Tensor` object,
which wraps all the usual arithmetic operations,
and other Torch functions.

## Tensor operations and `functional` API

Every operation
(on tensors, from `functional`, etc)
(wherever they are, e.g. in custom `Module.forward`)
must wrap to a `Module` with `create_returnn_layer_dict`.

## `Parameter`

The `Parameter` is just a derived `Tensor` with a special meaning.
In most cases, `Parameter`s are part of endpoint modules,
and thus don't need special logic,
as the endpoint modules are wrapped directly to RETURNN layers.

In case it is part of a non-endpoint module,
it would create a corresponding `VariableLayer` in RETURNN.

## Naming

In PyTorch, Python itself defines the names, as Python variables,
or attributes on an object,
and there is otherwise no explicit concept of names.

In PyTorch, when you serialize and save a module (`Module.state_dict`),
it recursively iterates through the module attributes,
and collects sub modules, parameters and buffers (tensors).
The keys in this state dict are defined by the attribute names.

In TensorFlow, every tensor and every operation
have an explicit name,
and there is a concept of name spaces,
to define these names in a hierarchical manner.

In RETURNN, every layer has an explicit name,
and this defines the TF namespace,
and thus all including parameters and operations
have the corresponding TF namespace.

When we convert a PyTorch module definition
to a RETURNN network,
we need to introduce RETURNN layer names.
This is tried to be straight-forward as you would expect it,
but can be a bit tricky in some cases.

Sub modules would use the corresponding attribute as name.

There is some special logic which determines
what would end up in the root namespace.

Temporary modules
(e.g. via the `functional` API, or tensor operations such as `x + y`)
would get a canonical name (e.g. `add`)
and be part of the RETURNN subnetwork
where they are created
(e.g. in a `Module.forward`).

There can be calculations in the `Module.__init__`,
e.g. to create a constant filter, or construct other constants,
or for custom parameter initialization, etc.
Currently, this ends up in a temporary namespace,
and will be evaluated on-the-fly (eagerly),
and will be used as constants later on (in `Module.forward`).

All the logic for the naming
is in [`naming`](../naming).


# Status

Obviously, this is incomplete.

For some status of what is not supported currently,
see [the unsupported document](../../Unsupported.md).

Otherwise, when you hit some `Module`
or `functional` function, or Tensor function
which is not implemented,
it just means that no-one has implemented it yet.

It already works fine for parts of [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN).
See the [`converter` documentation](../converter) for a full example.
