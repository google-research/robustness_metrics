# coding=utf-8
# Copyright 2020 The Robustness Metrics Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Vision Transformer (ViT) models.

The model architecture and training procedure are explained in:

  Dosovitskiy, Alexey, et al.
  "An image is worth 16x16 words: Transformers for image recognition at scale."
  arXiv preprint arXiv:2010.11929 (2020).


For more information on the available models and checkpoints, please refer to

https://github.com/google-research/vision_transformer

Note that to run this, you need vit_jax installed, which you can do using pip.

"""
# Adapted from
# https://github.com/google-research/vision_transformer/blob/master/vit_jax.ipynb
import jax
import numpy as onp
from robustness_metrics.common import pipeline_builder
from vit_jax import checkpoint
from vit_jax import models


def _pad_for_pmap(array):
  # Expands the batch_dim to [jax.device_count(), batch // jax.device_count]
  # padding as necessary.
  n = array.shape[0]
  new_dims = [jax.device_count(), -1] + list(array.shape[1:])
  padding_0 = (-n) % jax.device_count()
  padding = [(0, padding_0)] + [(0, 0) for _ in array.shape[1:]]
  return onp.pad(array, padding, "edge").reshape(new_dims)


# Example invocation:
#   model = "ViT-B_16"
#   resolution = 384
#   ckpt_path = gs://vit_models/imagenet21k+imagenet2012/ViT-B_16.npz
def create(model: str, resolution: int, ckpt_path: str):
  """Loads a ViT model from the given checkpoint.

  Args:
   model: The model name, for a list see vit_jax/models.KNOWN_MODELS.
   resolution: The image gets resized to `[resolution, resolution]`.
   ckpt_path: The .npz checkpoint file in the experiment's workdir.
  Returns:
    The model function and the corresponding preprocessing function.
  """
  # Assert that there is only one host!
  assert jax.host_count() == 1, "Multi-host setups not supported under JAX."

  try:
    model = models.KNOWN_MODELS[model].partial(num_classes=1000)
  except KeyError:
    raise ValueError(f"Unknown model {model!r}, available models: "
                     f"{list(models.KNOWN_MODELS)}")
  params = checkpoint.load(ckpt_path)
  params["pre_logits"] = {}  # Need to restore empty leaf for Flax.

  def _model_call(features, params):
    return jax.nn.softmax(model.call(params, features), axis=-1)

  model_call = jax.pmap(_model_call, static_broadcasted_argnums=[1])

  def call(features):
    images = features["image"].numpy()
    # We have to pad the images in case the batch_size is not divisibly by
    # the device count.
    images_for_pmap = _pad_for_pmap(images)
    pmap_result = model_call(images_for_pmap, params)
    n_images = images.shape[0]
    return pmap_result.reshape([-1] + list(pmap_result.shape[2:]))[:n_images]

  preprocess_config = f"resize({resolution})|value_range(-1,1)"
  preprocess_fn = pipeline_builder.get_preprocess_fn(
      preprocess_config, remove_tpu_dtypes=False)
  return call, preprocess_fn
