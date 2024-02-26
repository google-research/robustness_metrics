# coding=utf-8
# Copyright 2024 The Robustness Metrics Authors.
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

"""Define datasets for OOD detection tasks."""

import abc
from typing import Callable, Optional

from robustness_metrics.common import ops
from robustness_metrics.common import types
from robustness_metrics.datasets import base
from robustness_metrics.datasets import tfds as rm_tfds
import tensorflow as tf


def _set_label_to_one(feature):
  feature["label"] = tf.ones_like(feature["label"])
  feature["metadata"]["label"] = tf.ones_like(feature["label"])
  return feature


def _set_label_to_zero(feature):
  feature["label"] = tf.zeros_like(feature["label"])
  feature["metadata"]["label"] = tf.zeros_like(feature["label"])
  return feature


def _keep_common_fields(feature, spec):
  """Delete the keys of feature that are not in spec."""
  if not isinstance(feature, dict): return feature
  common_keys = set(feature.keys()) & set(spec.keys())
  return {
      key: _keep_common_fields(feature[key], spec[key]) for key in common_keys
  }


def _concatenate(in_ds: tf.data.Dataset,
                 out_ds: tf.data.Dataset,
                 ood_with_positive_labels: bool = False) -> tf.data.Dataset:
  """Concatenate in_ds and out_ds, making sure they have compatible specs."""
  in_spec = in_ds.element_spec
  out_spec = out_ds.element_spec

  def format_in_ds(feature):
    if ood_with_positive_labels:
      feature = _set_label_to_zero(feature)
    else:
      feature = _set_label_to_one(feature)
    return _keep_common_fields(feature, out_spec)

  def format_out_ds(feature):
    if ood_with_positive_labels:
      feature = _set_label_to_one(feature)
    else:
      feature = _set_label_to_zero(feature)
    return _keep_common_fields(feature, in_spec)

  return in_ds.map(format_in_ds).concatenate(out_ds.map(format_out_ds))


def _make_element_id_unique(dataset_tag: str):
  """We make element_id differ in the in- and out-of-distribution datasets."""
  dataset_fingerprint = ops.fingerprint_int64(dataset_tag)
  def _make_element_id_unique_fn(feature):
    fingerprint = (feature["metadata"]["element_id"], dataset_fingerprint)
    feature["metadata"]["element_id"] = ops.fingerprint_int64(fingerprint)
    return feature
  return _make_element_id_unique_fn


class OodDetectionDataset(base.Dataset, metaclass=abc.ABCMeta):
  """A dataset made of a pair of one in- and one out-of-distribution datasets.

  In this binary (detection) task, the in-distribution dataset typically has
  labels 1 and the out-of-distrbution dataset has labels 0. The other convention
  also exists (https://arxiv.org/pdf/1610.02136.pdf), which can be controlled by
  the argument `ood_with_positive_labels`.

  See https://arxiv.org/pdf/2106.03004.pdf for more background.
  """

  def __init__(self, ood_with_positive_labels: bool = False):
    """Initializes the OodDetectionDataset."""
    self._ood_with_positive_labels = ood_with_positive_labels

  @property
  def info(self) -> base.DatasetInfo:
    return base.DatasetInfo(num_classes=2)

  @property
  @abc.abstractmethod
  def in_dataset(self) -> base.Dataset:
    """The in-distribution dataset."""

  @property
  @abc.abstractmethod
  def out_dataset(self) -> base.Dataset:
    """The out-of-distribution dataset."""

  def load(self,
           preprocess_fn: Optional[Callable[[types.Features], types.Features]]
           ) -> tf.data.Dataset:

    in_ds = self.in_dataset.load(preprocess_fn)
    in_ds = in_ds.map(_make_element_id_unique("in_ds"))

    out_ds = self.out_dataset.load(preprocess_fn)
    out_ds = out_ds.map(_make_element_id_unique("out_ds"))

    return _concatenate(in_ds, out_ds, self._ood_with_positive_labels)


# The choices of the pairing of the datasets are motivated by the setting of:
#   https://arxiv.org/pdf/2106.03004.pdf, Appendix C.
@base.registry.register("cifar10_vs_cifar100")
class Cifar10VsCifar100Dataset(OodDetectionDataset):
  """The CIFAR-10 vs. CIFAR-100 ood detection dataset."""

  @property
  def in_dataset(self) -> base.Dataset:
    return rm_tfds.Cifar10Dataset()

  @property
  def out_dataset(self) -> base.Dataset:
    return rm_tfds.Cifar100Dataset()


@base.registry.register("cifar10_vs_dtd")
class Cifar10VsDtdDataset(OodDetectionDataset):
  """The CIFAR-10 vs. DTD ood detection dataset."""

  @property
  def in_dataset(self) -> base.Dataset:
    return rm_tfds.Cifar10Dataset()

  @property
  def out_dataset(self) -> base.Dataset:
    return rm_tfds.DtdDataset()


@base.registry.register("cifar10_vs_places365")
class Cifar10VsPlaces365Dataset(OodDetectionDataset):
  """The CIFAR-10 vs. Places365 ood detection dataset."""

  @property
  def in_dataset(self) -> base.Dataset:
    return rm_tfds.Cifar10Dataset()

  @property
  def out_dataset(self) -> base.Dataset:
    return rm_tfds.Places365Dataset()


@base.registry.register("cifar10_vs_svhn")
class Cifar10VsSvhnDataset(OodDetectionDataset):
  """The CIFAR-10 vs. SVHN ood detection dataset."""

  @property
  def in_dataset(self) -> base.Dataset:
    return rm_tfds.Cifar10Dataset()

  @property
  def out_dataset(self) -> base.Dataset:
    return rm_tfds.SvhnDataset()


@base.registry.register("cifar100_vs_cifar10")
class Cifar100VsCifar10Dataset(OodDetectionDataset):
  """The CIFAR-100 vs. CIFAR-10 ood detection dataset."""

  @property
  def in_dataset(self) -> base.Dataset:
    return rm_tfds.Cifar100Dataset()

  @property
  def out_dataset(self) -> base.Dataset:
    return rm_tfds.Cifar10Dataset()


@base.registry.register("cifar100_vs_dtd")
class Cifar100VsDtdDataset(OodDetectionDataset):
  """The CIFAR-100 vs. DTD ood detection dataset."""

  @property
  def in_dataset(self) -> base.Dataset:
    return rm_tfds.Cifar100Dataset()

  @property
  def out_dataset(self) -> base.Dataset:
    return rm_tfds.DtdDataset()


@base.registry.register("cifar100_vs_places365")
class Cifar100VsPlaces365Dataset(OodDetectionDataset):
  """The CIFAR-100 vs. Places365 ood detection dataset."""

  @property
  def in_dataset(self) -> base.Dataset:
    return rm_tfds.Cifar100Dataset()

  @property
  def out_dataset(self) -> base.Dataset:
    return rm_tfds.Places365Dataset()


@base.registry.register("cifar100_vs_svhn")
class Cifar100VsSvhnDataset(OodDetectionDataset):
  """The CIFAR-100 vs. SVHN ood detection dataset."""

  @property
  def in_dataset(self) -> base.Dataset:
    return rm_tfds.Cifar100Dataset()

  @property
  def out_dataset(self) -> base.Dataset:
    return rm_tfds.SvhnDataset()


# This choice of dataset pairing is motivated by the setting of:
#   https://arxiv.org/pdf/2207.07411.pdf.
@base.registry.register("imagenet_vs_places365")
class ImagenetVsPlaces365Dataset(OodDetectionDataset):
  """The Imagenet vs. Places365 ood detection dataset."""

  @property
  def in_dataset(self) -> base.Dataset:
    return rm_tfds.ImageNetDataset()

  @property
  def out_dataset(self) -> base.Dataset:
    return rm_tfds.Places365Dataset()
