This sub package provides utilities to convert
a PyTorch model
to a RETURNN model,
including a PyTorch checkpoint.

For the conversion, you provide some custom model function,
which would do some PyTorch computations.
This can potentially import other external PyTorch code.
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
    pyt_device = torch.device("cpu")
    generator = pwg_models.MelGANGenerator(**pwg_config['generator_params'])
    generator.load_state_dict(
        torch.load(args.pwg_checkpoint, map_location="cpu")["model"]["generator"])
    generator.remove_weight_norm()
    pwg_model = generator.eval().to(pyt_device)
    assert pwg_config["generator_params"].get("aux_context_window", 0) == 0  # not implemented otherwise
    pwg_pqmf = pwg_layers.PQMF(pwg_config["generator_params"]["out_channels"]).to(pyt_device)

    with torch.no_grad():
        return pwg_pqmf.synthesis(pwg_model(inputs))
```

The `wrapped_import` is like `importlib.import_module`
(without the `package` argument),
i.e. it imports a module.
The import is potentially wrapped
via our [import wrapper](../import_wrapper).

The transformation happens on multiple steps,
and we verify the outputs after each step.

(Note about randomness:
We reset the PyTorch and numpy random seed before each step.
In case this is non-deterministic,
this is ok as long as it stays within our numerical limits.)

Steps:

* First run the `model_func` as-is with no `wrapped_import`.
  This will run the original code, unmodified.
  This will result in our reference output.

* Now we use a `wrapped_import` which wraps `torch`
  to a traced variant of the original `torch` module.
  This should behave exactly the same,
  and we check that.
  It will only keep track of all `Module` creations,
  and keep track of individual inputs/outputs of module calls.
  This will use the [naming logic](../naming)
  to create uniquely create names.
  We later expect that when we our our `torch` module drop-in replacement,
  that we get exactly the same names.
  Then we can compare the outputs of individual modules.
  This will also keep track of parameters.

* Now use a `wrapped_import` which wraps `torch` 
  to our [`torch` drop-in replacement module](../torch).
  This will create a corresponding RETURNN network now.
  We check the naming to be consistent with the previous naming,
  and we also import the PyTorch parameters,
  and we run each RETURNN layer on the previous PyTorch inputs,
  and verify the outputs to the previous PyTorch outputs.
  We also dump the RETURNN net dict in the end,
  and store the TF checkpoint file.

* Now we create a RETURNN network based on the net dict,
  and then load the TF checkpoint file.
  Now we verify again whether we get the same final outputs.
