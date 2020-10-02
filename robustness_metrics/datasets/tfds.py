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
"""Wrappers for datasets in tfds."""
from typing import Any, Callable, Dict, Optional, Union

from robustness_metrics.common import ops
from robustness_metrics.common import pipeline_builder
from robustness_metrics.common import types
from robustness_metrics.datasets import base
import tensorflow as tf
import tensorflow_datasets as tfds

default_imagenet_preprocessing = None
default_config = "resize_small(256)|central_crop(224)|value_range(-1,1)"
default_imagenet_preprocessing = pipeline_builder.get_preprocess_fn(
    default_config, remove_tpu_dtypes=False)

PreprocessFn = Callable[[types.Features], types.Features]


class TFDSDataset(base.Dataset):
  """The base class of all `tensorflow_datasets` (TFDS) datasets.

  Two fields will be added to the wrapped dataset, before preprocessing it with
  the given function in `load` and batching. The two fields are:

  * `element_id`: A 64-bit integer identifying the element in the dataset by
    applying a fingerprint function to the field provided in the initializer.
  * `metadata`: A dictionary a single fields: `label`. If you want to add
    extra fields to the metadata, please override `create_metadata`.
  """

  def __init__(self,
               dataset_builder: tfds.core.DatasetBuilder,
               fingerprint_key: str,
               split: Union[str, tfds.Split] = "test",
               label_key: Optional[str] = "label",
               default_preprocess_fn: Optional[PreprocessFn] = None):
    """Initializes the object.

    Args:
      dataset_builder: The tfds builder for the dataset.
      fingerprint_key: The name of the feature holding a string that will be
        used to create an element id using a fingerprinting function.
      split: The name of the dataset split.
      label_key: The name of the field holding the label.
      default_preprocess_fn: The function used to preprocess the data in `load`
        if no function is provided there.
    """
    self._dataset_builder = dataset_builder
    self._fingerprint_key = fingerprint_key
    self._split = split
    self._label_key = label_key
    self._default_preprocess_fn = default_preprocess_fn

  @property
  def info(self) -> base.DatasetInfo:
    if self._label_key:
      label_feature = self._dataset_builder.info.features[self._label_key]
      return base.DatasetInfo(num_classes=label_feature.num_classes)
    else:
      return base.DatasetInfo(num_classes=None)

  def create_metadata(self, features):
    features["metadata"] = {
        "label": features[self._label_key],
    }
    return features

  def load(self, preprocess_fn: Optional[PreprocessFn],
           batch_size: int) -> tf.data.Dataset:
    if not preprocess_fn:
      preprocess_fn = self._default_preprocess_fn

    def create_element_id(features: Dict[str, Any]):
      """Hash the element id to compute a unique id."""
      assert "element_id" not in features, \
             "`element_id` should not be already present in the feature set."
      fingerprint_feature = features[self._fingerprint_key]
      features["element_id"] = ops.fingerprint_int64(fingerprint_feature)
      return features

    preprocess_fn = ops.compose(preprocess_fn, create_element_id,
                                self.create_metadata)
    self._dataset_builder.download_and_prepare()
    ds = self._dataset_builder.as_dataset(
        split=self._split, as_supervised=False)
    ds_batched = ds.map(preprocess_fn).batch(batch_size, drop_remainder=False)
    return ds_batched.prefetch(tf.data.experimental.AUTOTUNE)


@base.registry.register("imagenet")
class ImageNetDataset(TFDSDataset):
  """The ImageNet validation set."""

  def __init__(self):
    super().__init__(
        tfds.builder("imagenet2012"),
        "file_name",
        "validation",
        default_preprocess_fn=default_imagenet_preprocessing)


@base.registry.register("cifar10")
class Cifar10Dataset(TFDSDataset):
  """The CIFAR-10 dataset."""

  def __init__(self):
    super().__init__(dataset_builder=tfds.builder("cifar10"),
                     fingerprint_key="id",
                     default_preprocess_fn=default_cifar_preprocessing)


@base.registry.register("cifar10_c")
class Cifar10CDataset(TFDSDataset):
  """The CIFAR10-C dataset."""

  def __init__(self, corruption_type, severity):
    tfds_variant_name = f"cifar10_corrupted/{corruption_type}_{severity}"
    super().__init__(dataset_builder=tfds.builder(tfds_variant_name),
                     fingerprint_key="",
                     default_preprocess_fn=default_cifar_preprocessing)

  def load(self, preprocess_fn: Optional[PreprocessFn],
           batch_size: int) -> tf.data.Dataset:
    if not preprocess_fn:
      preprocess_fn = self._default_preprocess_fn

    preprocess_fn = ops.compose(preprocess_fn, self.create_metadata)
    ds = self._dataset_builder.as_dataset(split=self._split,
                                          as_supervised=False)
    # TODO(trandustin): Change to drop_remainder=False. For now, True aligns
    # with how results are currently measured in Uncertainty Baselines.
    ds_batched = ds.map(preprocess_fn).batch(batch_size, drop_remainder=True)
    return ds_batched.prefetch(tf.data.experimental.AUTOTUNE)


@base.registry.register("cifar100")
class Cifar100Dataset(TFDSDataset):
  """The CIFAR-100 dataset."""

  def __init__(self):
    super().__init__(dataset_builder=tfds.builder("cifar100"),
                     fingerprint_key="id",
                     default_preprocess_fn=default_cifar_preprocessing)


@base.registry.register("imagenet_a")
class ImageNetADataset(TFDSDataset):
  """The ImageNet-A dataset."""

  def __init__(self):
    super().__init__(
        tfds.builder("imagenet_a"),
        "file_name",
        "test",
        default_preprocess_fn=default_imagenet_preprocessing)


@base.registry.register("imagenet_r")
class ImageNetRDataset(TFDSDataset):
  """The ImageNet-R dataset."""

  def __init__(self):
    super().__init__(
        tfds.builder("imagenet_r"),
        "file_name",
        "test",
        default_preprocess_fn=default_imagenet_preprocessing)


@base.registry.register("imagenet_v2")
class ImageNetV2Dataset(TFDSDataset):
  """The ImageNet-V2 dataset."""

  def __init__(self, variant):
    tfds_variant_name = {
        "MATCHED_FREQUENCY": "matched-frequency",
        "TOP_IMAGES": "topimages",
        "THRESHOLDED": "threshold-0.7",
    }[variant]
    super().__init__(
        tfds.builder(f"imagenet_v2/{tfds_variant_name}"),
        "file_name",
        "test",
        default_preprocess_fn=default_imagenet_preprocessing)


@base.registry.register("imagenet_c")
class ImageNetCDataset(TFDSDataset):
  """The ImageNet-C dataset."""

  def __init__(self, corruption_type, severity):
    tfds_variant_name = f"imagenet2012_corrupted/{corruption_type}_{severity}"
    super().__init__(
        tfds.builder(tfds_variant_name),
        "file_name",
        "validation",
        default_preprocess_fn=default_imagenet_preprocessing)


def default_cifar_preprocessing(features: types.Features) -> types.Features:
  """Applies mean/std normalization to images."""
  dtype = tf.float32
  image = features["image"]
  image = tf.image.convert_image_dtype(image, dtype)
  mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=dtype)
  std = tf.constant([0.2023, 0.1994, 0.2010], dtype=dtype)
  features["image"] = (image - mean) / std
  return features
