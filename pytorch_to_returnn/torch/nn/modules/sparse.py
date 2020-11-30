
from __future__ import annotations
import math
from typing import Dict, Any, Optional
import tensorflow as tf
from returnn.tf.layers.basic import LinearLayer
from .module import Module
from ..parameter import Parameter
from ...tensor import Tensor
from .. import init


class Embedding(Module):

  def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
               max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
               sparse: bool = False, _weight: Optional[Tensor] = None) -> None:
    super(Embedding, self).__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    if padding_idx is not None:
      if padding_idx > 0:
        assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
      elif padding_idx < 0:
        assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
        padding_idx = self.num_embeddings + padding_idx
    self.padding_idx = padding_idx
    self.max_norm = max_norm
    self.norm_type = norm_type
    self.scale_grad_by_freq = scale_grad_by_freq
    if _weight is None:
      self.weight = Parameter(Tensor(num_embeddings, embedding_dim))
      self.reset_parameters()
    else:
      assert list(_weight.shape) == [num_embeddings, embedding_dim], (
        'Shape of weight does not match num_embeddings and embedding_dim')
      self.weight = Parameter(_weight)
    self.sparse = sparse

  def reset_parameters(self) -> None:
    init.normal_(self.weight)
    if self.padding_idx is not None:
      self.weight[self.padding_idx].fill_(0)

  @classmethod
  def from_pretrained(cls, embeddings, freeze=True, padding_idx=None,
                      max_norm=None, norm_type=2., scale_grad_by_freq=False,
                      sparse=False):
    assert embeddings.dim() == 2, 'Embeddings parameter is expected to be 2-dimensional'
    rows, cols = embeddings.shape
    embedding = cls(
      num_embeddings=rows,
      embedding_dim=cols,
      _weight=embeddings,
      padding_idx=padding_idx,
      max_norm=max_norm,
      norm_type=norm_type,
      scale_grad_by_freq=scale_grad_by_freq,
      sparse=sparse)
    embedding.weight.requires_grad = not freeze
    return embedding

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    return {
      "class": "linear", "from": self._get_input_layer_name(input),
      "activation": None, "with_bias": False,
      "n_out": self.embedding_dim}

  def check_returnn_layer(self, layer: LinearLayer):
    return layer.input_data.sparse and layer.input_data.dim == self.num_embeddings

  def import_params_torch_to_returnn(self, *, layer: LinearLayer, torch_module: Embedding):
    session = tf.compat.v1.get_default_session()
    values = torch_module.weight.detach().numpy()
    layer.params["W"].load(values, session=session)


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
