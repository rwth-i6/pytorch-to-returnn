
from typing import Set


class WrapCtx:
  def __init__(self, wrapped_mod_prefix: str, wrap_mods_direct: Set[str], wrap_mods_indirect: Set[str]):
    """
    :param wrapped_mod_prefix: e.g. "pytorch_to_returnn._wrapped_mods."
    :param wrap_mods_direct: e.g. {"torch.nn.modules"}
      These modules will be imported using the direct AST transformation,
      and result in :class:`WrappedSourceModule`.
    :param wrap_mods_indirect: e.g. {"torch", "torch.nn.modules.module"}
      These modules will be wrapped using :class:`WrappedIndirectModule`,
      which wraps the originally unmodified imported module.
      There can be overlaps between wrap_mods_direct and wrap_mods_indirect.
      The topmost entry will decide whether it is direct or indirect.
    """
    assert wrapped_mod_prefix.endswith(".")
    self.wrapped_mod_prefix = wrapped_mod_prefix
    self.wrap_mods_direct = wrap_mods_direct
    self.wrap_mods_indirect = wrap_mods_indirect

    #_KeepAsIsTypes


def make_torch_default_ctx(wrapped_mod_prefix: str) -> WrapCtx:
  return WrapCtx(
    wrapped_mod_prefix=wrapped_mod_prefix,
  )


_ExplicitIndirectModList = {
  # Leave "torch" in to not do any magic on the original Torch modules.
  # Note that the main torch module cannot be transformed
  # because of some side-effects triggered by native Torch code,
  # which we cannot catch automatically.
  # More specifically, in torch/__init__.py, there is the call _C._initExtension,
  # which (un-intuitively) does all sorts of initialization,
  # including torch::utils::initializeDtypes,
  # which itself has the side-effect of adding dtypes (e.g. `float32`)
  # to the `torch` module (-> `torch.float32`).
  # And the name `"torch"` is hardcoded there,
  # i.e. it will not used our wrapped module,
  # thus we will never have `float32` registered,
  # and that leads to exceptions.
  # This is likely not the only problem.
  "torch",
}
if "torch" in _ExplicitIndirectModList:
  _ExplicitIndirectModList.update({
    # If we would transform `torch` directly, the following should be explicitly excluded.
    "torch.tensor", "torch.distributed",
    "torch._C",
    # Avoid to have the `Module` class twice.
    "torch.nn.modules.module",
  })

_ExplicitDirectModList = {
  # Note: When we don't wrap all of `torch` (i.e. `"torch"` is in _ExplicitIndirectModList),
  # this will partially be WrappedIndirectModule and WrappedSourceModule.
  # This is not a problem in principle, but there can be subtle problems.
  # Also the import order can be important.
  # E.g. "torch.nn.modules" should be imported before "torch.nn.functional".
  # This might need extra explicit handling, depending what we have here in this list.
  "torch.nn.modules",
  # "torch.nn.functional",  # -- not needed, also causes other problems
}
