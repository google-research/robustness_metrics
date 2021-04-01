# coding=utf-8
# Copyright 2021 The Robustness Metrics Authors.
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
"""Model from `Big Transfer (BiT): General Visual Representation Learning`.

The model is pre-trained on ImageNet21k and fine-tuned to ImageNet
(https://arxiv.org/abs/1912.11370). The `logits` key in `representation`
signature contains the ImageNet logits.
"""
from robustness_metrics.common import pipeline_builder
import tensorflow as tf
import tensorflow_hub as hub


def create(dataset, network, size, resolution=None):
  """Returns the BiT pre-trained models with the linear classifier layer."""
  # Mapping the dataset names to the BiT qualifiers.
  supported_datasets = {
      "Imagenet1k": "s",
      "Imagenet21k": "m",
  }
  supported_networks = ["R50", "R101", "R152"]
  supported_sizes = ["x1", "x3", "x4"]

  if size not in supported_sizes:
    raise ValueError(f"Size {size} is not in {supported_sizes!r}.")
  if dataset not in supported_datasets:
    raise ValueError(f"Dataset {dataset} is not in {supported_datasets!r}.")
  if network not in supported_networks:
    raise ValueError(f"Network {network} is not in {supported_networks!r}.")

  root = ("https://tfhub.dev/google/bit/{qualifier}-{network_name}{size}/"
          "ilsvrc2012_classification/1")
  path = root.format(
      qualifier=supported_datasets[dataset],
      network_name=network.lower(),
      size=size)
  module = hub.load(path)

  @tf.function
  def model(features):
    return tf.nn.softmax(module(features["image"]), axis=-1)

  if resolution is not None:
    preprocess_config_fmt = "resize_small({})|central_crop({})|value_range(0,1)"
    preprocess_config = preprocess_config_fmt.format(
        int(1.15 * resolution), resolution)
  elif size == "x4":
    preprocess_config = "resize_small(512)|central_crop(480)|value_range(0,1)"
  else:
    preprocess_config = "resize(384)|value_range(0,1)"
  preprocess_fn = pipeline_builder.get_preprocess_fn(
      preprocess_config, remove_tpu_dtypes=False)

  return model, preprocess_fn
