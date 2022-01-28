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


def _enumerated_to_metadata(position, features):
  features["metadata"]["element_id"] = tf.reshape(position, [1])
  return features


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
               fingerprint_key: Optional[str] = None,
               split: Union[str, tfds.Split] = "test",
               label_key: Optional[str] = "label",
               default_preprocess_fn: Optional[PreprocessFn] = None):
    """Initializes the object.

    Args:
      dataset_builder: The tfds builder for the dataset.
      fingerprint_key: The name of the feature holding a string that will be
        used to create an element id using a fingerprinting function. If it is
        equal to None, the logic for `create_metadata` has to be overriden.
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

  def _compute_element_id(self, features: Dict[str, Any]):
    """Hash the element id to compute a unique id."""
    assert_msg = "`element_id` should not be present in the feature set."
    assert "element_id" not in features, assert_msg
    fingerprint_feature = features[self._fingerprint_key]
    return ops.fingerprint_int64(fingerprint_feature)

  def create_metadata(self, features):
    if self._fingerprint_key is None:
      error_msg = ("If fingerprint_key=None, the logic of `create_metadata` has"
                   " to be overriden.")
      raise NotImplementedError(error_msg)
    features["metadata"] = {
        "label": features[self._label_key],
        "element_id": self._compute_element_id(features),
    }
    return features

  def load(self,
           preprocess_fn: Optional[PreprocessFn] = None) -> tf.data.Dataset:
    if not preprocess_fn:
      preprocess_fn = self._default_preprocess_fn

    preprocess_fn = ops.compose(preprocess_fn,
                                self.create_metadata)
    self._dataset_builder.download_and_prepare()
    ds = self._dataset_builder.as_dataset(
        split=self._split, as_supervised=False)
    return ds.map(preprocess_fn)


@base.registry.register("imagenet")
class ImageNetDataset(TFDSDataset):
  """The ImageNet validation set."""

  def __init__(self, split="validation"):
    super().__init__(
        tfds.builder("imagenet2012"),
        "file_name",
        split=split,
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
                     fingerprint_key="_SHOULD_NOT_BE_USED",
                     default_preprocess_fn=default_cifar_preprocessing)

  def create_metadata(self, features):
    features["metadata"] = {
        "label": features[self._label_key],
    }
    return features

  def load(self, preprocess_fn: Optional[PreprocessFn]) -> tf.data.Dataset:
    if not preprocess_fn:
      preprocess_fn = self._default_preprocess_fn

    preprocess_fn = ops.compose(preprocess_fn, self.create_metadata)
    ds = self._dataset_builder.as_dataset(split=self._split,
                                          as_supervised=False)
    ds = ds.map(preprocess_fn)
    return ds.enumerate().map(_enumerated_to_metadata)


@base.registry.register("cifar100")
class Cifar100Dataset(TFDSDataset):
  """The CIFAR-100 dataset."""

  def __init__(self):
    super().__init__(dataset_builder=tfds.builder("cifar100"),
                     fingerprint_key="id",
                     default_preprocess_fn=default_cifar_preprocessing)


@base.registry.register("oxford_flowers102")
class OxfordFlowers102Dataset(TFDSDataset):
  """The oxford_flowers102 dataset.

  TFDS page: https://www.tensorflow.org/datasets/catalog/oxford_flowers102
  Original page: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
  """

  def __init__(self):
    super().__init__(dataset_builder=tfds.builder("oxford_flowers102"),
                     fingerprint_key="file_name",
                     default_preprocess_fn=default_imagenet_preprocessing)


@base.registry.register("oxford_iiit_pet")
class OxfordIiitPetDataset(TFDSDataset):
  """The oxford_iiit_pet dataset.

  We only keep the 'image', 'label' and 'file_name' fields, the last one being
  used for the fingerprint_key.

  TFDS page: https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet
  Original page: http://www.robots.ox.ac.uk/~vgg/data/pets/
  """

  def __init__(self):
    super().__init__(dataset_builder=tfds.builder("oxford_iiit_pet"),
                     fingerprint_key="file_name",
                     default_preprocess_fn=default_imagenet_preprocessing)

  def load(self,
           preprocess_fn: Optional[PreprocessFn] = None) -> tf.data.Dataset:
    ds = super().load(preprocess_fn)

    def delete_useless_fields(feature):
      del feature["segmentation_mask"]
      del feature["species"]
      return feature

    return ds.map(delete_useless_fields)


@base.registry.register("places365")
class Places365Dataset(TFDSDataset):
  """The places365_small dataset.

  Only 'image' and 'label' are available; fingerprint_key based on the position
  of the element in the dataset.

  TFDS page: https://www.tensorflow.org/datasets/catalog/places365_small
  Original page: http://places2.csail.mit.edu/
  """

  def __init__(self):
    super().__init__(dataset_builder=tfds.builder("places365_small"),
                     default_preprocess_fn=default_imagenet_preprocessing)

  def create_metadata(self, features):
    features["metadata"] = {
        "label": features[self._label_key],
    }
    return features

  def load(self,
           preprocess_fn: Optional[PreprocessFn] = None) -> tf.data.Dataset:
    ds = super().load(preprocess_fn)
    return ds.enumerate().map(_enumerated_to_metadata)


@base.registry.register("dtd")
class DtdDataset(TFDSDataset):
  """The Describable Textures Dataset (DTD) dataset.

  TFDS page: https://www.tensorflow.org/datasets/catalog/dtd
  Original page: https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html
  """

  def __init__(self):
    super().__init__(dataset_builder=tfds.builder("dtd"),
                     fingerprint_key="file_name",
                     default_preprocess_fn=default_imagenet_preprocessing)


@base.registry.register("svhn")
class SvhnDataset(TFDSDataset):
  """The Street View House Numbers (SVHN) dataset.

  Only 'image' and 'label' are available; fingerprint_key based on the position
  of the element in the dataset.

  TFDS page: https://www.tensorflow.org/datasets/catalog/svhn_cropped
  Original page: http://ufldl.stanford.edu/housenumbers/
  """

  def __init__(self):
    super().__init__(dataset_builder=tfds.builder("svhn_cropped"),
                     default_preprocess_fn=default_imagenet_preprocessing)

  def create_metadata(self, features):
    features["metadata"] = {
        "label": features[self._label_key],
    }
    return features

  def load(self,
           preprocess_fn: Optional[PreprocessFn] = None) -> tf.data.Dataset:
    ds = super().load(preprocess_fn)
    return ds.enumerate().map(_enumerated_to_metadata)


_IMAGENET_A_LABELSET = [
    6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79,
    89, 90, 94, 96, 97, 99, 105, 107, 108, 110, 113, 124, 125, 130, 132, 143,
    144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295, 298, 301, 306,
    307, 308, 309, 310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330,
    334, 335, 336, 347, 361, 363, 372, 378, 386, 397, 400, 401, 402, 404, 407,
    411, 416, 417, 420, 425, 428, 430, 437, 438, 445, 456, 457, 461, 462, 470,
    472, 483, 486, 488, 492, 496, 514, 516, 528, 530, 539, 542, 543, 549, 552,
    557, 561, 562, 569, 572, 573, 575, 579, 589, 606, 607, 609, 614, 626, 627,
    640, 641, 642, 643, 658, 668, 677, 682, 684, 687, 701, 704, 719, 736, 746,
    749, 752, 758, 763, 765, 768, 773, 774, 776, 779, 780, 786, 792, 797, 802,
    803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845, 847, 850, 859, 862,
    870, 879, 880, 888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937, 943,
    945, 947, 951, 954, 956, 957, 959, 971, 972, 980, 981, 984, 986, 987, 988,
]


@base.registry.register("imagenet_a")
class ImageNetADataset(TFDSDataset):
  """The ImageNet-A dataset."""

  @property
  def info(self) -> base.DatasetInfo:
    return base.DatasetInfo(num_classes=super().info.num_classes,
                            appearing_classes=_IMAGENET_A_LABELSET)

  def __init__(self):
    super().__init__(
        tfds.builder("imagenet_a"),
        "file_name",
        "test",
        default_preprocess_fn=default_imagenet_preprocessing)


_IMAGENET_R_LABELSET = [
    1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84,
    90, 94, 96, 97, 99, 100, 105, 107, 113, 122, 125, 130, 132, 144, 145, 147,
    148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203,
    207, 208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260,
    263, 265, 267, 269, 276, 277, 281, 288, 289, 291, 292, 293, 296, 299, 301,
    308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340,
    341, 344, 347, 353, 355, 361, 362, 365, 366, 367, 368, 372, 388, 390, 393,
    397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447, 448, 457, 462,
    463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583,
    587, 593, 594, 596, 609, 613, 617, 621, 629, 637, 657, 658, 701, 717, 724,
    763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852,
    866, 875, 883, 889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945,
    947, 948, 949, 951, 953, 954, 957, 963, 965, 967, 980, 981, 983, 988,
]


@base.registry.register("imagenet_r")
class ImageNetRDataset(TFDSDataset):
  """The ImageNet-R dataset."""

  @property
  def info(self) -> base.DatasetInfo:
    return base.DatasetInfo(num_classes=super().info.num_classes,
                            appearing_classes=_IMAGENET_R_LABELSET)

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

  def __init__(self, corruption_type, severity, split="validation"):
    tfds_variant_name = f"imagenet2012_corrupted/{corruption_type}_{severity}"
    super().__init__(
        tfds.builder(tfds_variant_name),
        "file_name",
        split=split,
        default_preprocess_fn=default_imagenet_preprocessing)


@base.registry.register("synthetic")
class SyntheticData(TFDSDataset):
  """A dataset of foreground objects pasted on random backgrounds."""

  def __init__(self, variant):
    if variant not in ["size", "rotation", "location"]:
      raise ValueError(
          f"Variant {variant} not in ['size', 'rotation', 'location']")

    self.variant = variant
    tfds_variant_name = f"siscore/{variant}"
    super().__init__(dataset_builder=tfds.builder(tfds_variant_name),
                     fingerprint_key="image_id",
                     split="test",
                     default_preprocess_fn=default_imagenet_preprocessing)

  def create_metadata(self, features):
    features["metadata"] = {
        "label": features[self._label_key],
        "element_id": features["image_id"],
        "image_id": features["image_id"],
        "dataset_variant": self.variant,
    }
    return features


def default_cifar_preprocessing(features: types.Features) -> types.Features:
  """Applies mean/std normalization to images."""
  dtype = tf.float32
  image = features["image"]
  image = tf.image.convert_image_dtype(image, dtype)
  mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=dtype)
  std = tf.constant([0.2023, 0.1994, 0.2010], dtype=dtype)
  features["image"] = (image - mean) / std
  return features


@base.registry.register("imagenet_sketch")
class ImageNetSketchDataset(TFDSDataset):
  """The ImageNet-Sketch Dataset."""

  def __init__(self):
    super().__init__(dataset_builder=tfds.builder("imagenet_sketch"),
                     fingerprint_key="file_name",
                     default_preprocess_fn=default_imagenet_preprocessing)
