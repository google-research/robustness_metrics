# coding=utf-8
# Copyright 2022 The Robustness Metrics Authors.
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
"""Simple Framework for Contrastive Learning of Visual Representations.

The model is trained on ImageNet (https://arxiv.org/abs/2002.05709). The default
signature contains the `logits_sup` key which corresponds to ImageNet logits.
Here are the available models with the ImageNet Top-1 performance:
"""

from robustness_metrics.common import pipeline_builder
import tensorflow as tf
import tensorflow_hub as hub
# Please add simclr from https://github.com/google-research/simclr to the
# environment variable PYTHONPATH so that the following import works.
from simclr import data_util


ROOT = "gs://simclr-checkpoints/simclrv1/{version}/{size}/hub"
MODULE_PATHS = {
    "1x-self-supervised": ROOT.format(version="pretrain", size="1x"),
    "2x-self-supervised": ROOT.format(version="pretrain", size="2x"),
    "4x-self-supervised": ROOT.format(version="pretrain", size="4x"),

    "1x-fine-tuned-10": ROOT.format(version="finetune_10pct", size="1x"),
    "2x-fine-tuned-10": ROOT.format(version="finetune_10pct", size="2x"),
    "4x-fine-tuned-10": ROOT.format(version="finetune_10pct", size="4x"),

    "1x-fine-tuned-100": ROOT.format(version="finetune_100pct", size="1x"),
    "2x-fine-tuned-100": ROOT.format(version="finetune_100pct", size="2x"),
    "4x-fine-tuned-100": ROOT.format(version="finetune_100pct", size="4x"),
}



def preprocess_fn_default(features):
  features["image"] = data_util.preprocess_image(
      features["image"], 224, 224,
      is_training=False,
      color_distort=False,
      test_crop=True)
  return features


def create(size="1x", variant="self-supervised", resolution=None):
  """Returns the SimCLR pre-trained models with the linear classifier layer.

  Args:
    size: Width of the ResNet-50 model in ('1x', '2x', '4x').
    variant: String in ["self-supervised", "fine-tuned-100"].
    resolution: If set, the preprocessing function will first 1) crop the
      smaller side to `1.15 * resolution`, and then take a square central crop
      of size `resolution`.
  Returns:
    tf.function wrapping the model.
  """
  supported_sizes = ["1x", "2x", "4x"]
  supported_variants = ["self-supervised", "fine-tuned-10", "fine-tuned-100"]

  if size not in supported_sizes:
    raise ValueError(f"Size {size} is not in {supported_sizes!r}.")
  if variant not in supported_variants:
    raise ValueError(f"Variant {variant} is not in {supported_variants!r}.")

  module_id = f"{size}-{variant}"
  module_path = MODULE_PATHS[module_id]
  # The default signature is the 2048 dimensional representation, however we
  # want the ImageNet logits here.
  module = hub.KerasLayer(module_path, output_key="logits_sup")

  @tf.function
  def model(features):
    return tf.nn.softmax(module(features["image"]), axis=-1)

  if resolution is not None:
    preprocess_config_fmt = "resize_small({})|central_crop({})|value_range(0,1)"
    preprocess_config = preprocess_config_fmt.format(
        int(1.15 * resolution), resolution
    )
    preprocess_fn = pipeline_builder.get_preprocess_fn(
        preprocess_config, remove_tpu_dtypes=False)
  else:
    preprocess_fn = preprocess_fn_default

  return model, preprocess_fn
