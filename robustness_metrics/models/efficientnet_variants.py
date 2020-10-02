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
"""EfficientNet models pre-trained with different training protocols.

EfficientNet: https://arxiv.org/abs/1805.09501
AutoAugment: https://arxiv.org/abs/1805.09501
Adv-Prop: https://arxiv.org/abs/1911.09665
Noisy Student: https://arxiv.org/abs/1911.04252

This code uses TF-Hub modules derived from the checkpoints at

https://github.com/tensorflow/tpu/tree/4572efe4a9f709d01dbee85f948218af8b05a36c/models/official/efficientnet
"""

from robustness_metrics.common import pipeline_builder
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_hub as hub

CROP_PADDING = 32
EFFICIENTNET_RESOLUTIONS = [224, 240, 260, 300, 380, 456, 528, 600]


GS_ROOT = "gs://cloud-tpu-checkpoints/efficientnet/tfhub"
MODEL_PATHS = {
    # EfficientNet trained on ImageNet.
    "std": GS_ROOT + "/base/efficientnet-b{}",
    # EfficientNet trained on ImageNet with AutoAugment.
    "aa": GS_ROOT + "/aa/efficientnet-b{}",
    # EfficientNet trained on ImageNet with Adv-Prop.
    "adv-prop": GS_ROOT + "/adv/efficientnet-b{}",
    # EfficientNet trained on ImageNet/JFT with the Noisy Student protocol.
    "noisy-student": GS_ROOT + "/ns/efficientnet-b{}",
}



def create(model_index=0, variant="std", resolution=None):
  """Create EfficientNet models with corresponding preprocessing operations."""

  if variant not in ("std", "aa", "adv-prop", "noisy-student"):
    raise ValueError(f"EfficientNet variant not supported: {variant}")

  # Note that for the standard EfficientNet variant only B0-B5 architectures are
  # supported, and B0-B7 for all other variants.
  if (variant == "std" and model_index not in range(6)) \
     or (variant != "std" and model_index not in range(8)):
    raise ValueError(
        f"Invalid `model_index` {model_index} for EfficientNet `variant` "
        f"{variant}!")

  noisy_student = hub.KerasLayer(MODEL_PATHS[variant].format(model_index))

  @tf.function
  def model(features):
    images = features["image"]
    return tf.nn.softmax(noisy_student(images), axis=-1)

  def preprocess_fn(features):
    # EfficientNet preprocessing with model-dependent input resolution.
    # Preprocessing mimicks that of the public EfficientNet code from
    # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/preprocessing.py
    # (both `_resize_image` and `_decode_and_center_crop` taken from that code)

    def _resize_image(image, image_size, method=None):
      if method is not None:
        return tf1.image.resize([image], [image_size, image_size], method)[0]
      return tf1.image.resize_bicubic([image], [image_size, image_size])[0]

    def _decode_and_center_crop(image, image_size, resize_method=None):
      """Crops to center of image with padding then scales image_size."""
      shape = tf1.shape(image)
      image_height = shape[0]
      image_width = shape[1]

      padded_center_crop_size = tf1.cast(
          ((image_size / (image_size + CROP_PADDING)) *
           tf.cast(tf.minimum(image_height, image_width), tf.float32)),
          tf.int32)

      offset_height = ((image_height - padded_center_crop_size) + 1) // 2
      offset_width = ((image_width - padded_center_crop_size) + 1) // 2
      image = tf1.image.crop_to_bounding_box(image, offset_height, offset_width,
                                             padded_center_crop_size,
                                             padded_center_crop_size)
      image = _resize_image(image, image_size, resize_method)
      return image

    features["image"] = _decode_and_center_crop(
        features["image"], EFFICIENTNET_RESOLUTIONS[model_index])
    features["image"] = tf1.cast(features["image"], tf1.float32)
    # We assume the modules expect pixels in [-1, 1].
    features["image"] = features["image"] / 127.5 - 1.0

    return features

  if resolution is not None:
    preprocess_config_fmt = "resize_small({})|central_crop({})|value_range(-1,1)"
    preprocess_config = preprocess_config_fmt.format(
        int(1.15 * resolution), resolution)
    preprocess_fn = pipeline_builder.get_preprocess_fn(
        preprocess_config, remove_tpu_dtypes=False)

  return model, preprocess_fn
