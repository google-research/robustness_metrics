# coding=utf-8
# Copyright 2023 The Robustness Metrics Authors.
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

"""Tests for OOD detection datasets."""

from absl.testing import parameterized

import robustness_metrics as rm
from robustness_metrics.datasets import tfds as rm_tfds
import tensorflow as tf
import tensorflow_datasets as tfds

C10 = rm_tfds.Cifar10Dataset
C100 = rm_tfds.Cifar100Dataset
ImageNet = rm_tfds.ImageNetDataset
NUM_EXAMPLES = 32


class OodDetectionDatasetsTest(parameterized.TestCase, tf.test.TestCase):

  necessary_fields = ["image", "metadata"]

  @parameterized.parameters([
      # cifar10 versus *
      ("cifar10_vs_cifar100", [32, 32, 3]),
      ("cifar10_vs_dtd", [224, 224, 3]),
      ("cifar10_vs_places365", [224, 224, 3]),
      ("cifar10_vs_svhn", [224, 224, 3]),
      ("cifar10_vs_cifar100(ood_with_positive_labels=True)", [32, 32, 3]),
      ("cifar10_vs_dtd(ood_with_positive_labels=True)", [224, 224, 3]),
      ("cifar10_vs_places365(ood_with_positive_labels=True)", [224, 224, 3]),
      ("cifar10_vs_svhn(ood_with_positive_labels=True)", [224, 224, 3]),
      # cifar100 versus *
      ("cifar100_vs_cifar10", [32, 32, 3]),
      ("cifar100_vs_dtd", [224, 224, 3]),
      ("cifar100_vs_places365", [224, 224, 3]),
      ("cifar100_vs_svhn", [224, 224, 3]),
      ("cifar100_vs_cifar10(ood_with_positive_labels=True)", [32, 32, 3]),
      ("cifar100_vs_dtd(ood_with_positive_labels=True)", [224, 224, 3]),
      ("cifar100_vs_places365(ood_with_positive_labels=True)", [224, 224, 3]),
      ("cifar100_vs_svhn(ood_with_positive_labels=True)", [224, 224, 3]),
      # imagenet versus *
      ("imagenet_vs_places365", [224, 224, 3]),
      ("imagenet_vs_places365(ood_with_positive_labels=True)", [224, 224, 3]),
  ])
  def test_that_it_loads_with_default(self,
                                      ds_name,
                                      out_ds_expected_shape,
                                      label_field="label"):
    batch_size = 8
    assert_msg = ("The unit test requires NUM_EXAMPLES to be a multiple of the "
                  f" batch_size; received {NUM_EXAMPLES} and {batch_size}.")
    assert NUM_EXAMPLES % batch_size == 0, assert_msg

    if "imagenet_vs" in ds_name:
      in_ds_expected_shape = [batch_size] + [224, 224, 3]
    else:
      in_ds_expected_shape = [batch_size] + [32, 32, 3]
    out_ds_expected_shape = [batch_size] + out_ds_expected_shape

    # The in- and out-of-distribution datasets are concatenated, with a total
    # size equal to 2 * NUM_EXAMPLES.
    dataset = rm.datasets.get(ds_name)
    dataset = dataset.load(preprocess_fn=None)
    self.assertEqual(dataset.cardinality(), 2 * NUM_EXAMPLES)

    dataset = dataset.batch(batch_size)
    for batch_index, features in enumerate(dataset, 1):
      for feature in self.necessary_fields + [label_field]:
        self.assertIn(feature, features.keys())
      self.assertEqual(features["metadata"]["element_id"].dtype, tf.int64)
      # The first NUM_EXAMPLES datapoints are the in-distribution datapoints.
      # Their labels are 1's. Conversely, the last NUM_EXAMPLES datapoints are
      # the out-of-distribution datapoints, with labels equal to 0's.
      # The definition of the 1's and 0's is reversed when the argument
      # `ood_with_positive_labels` is set to True.
      in_label = 0 if "ood_with_positive_labels=True" in ds_name else 1
      ood_label = 1 if "ood_with_positive_labels=True" in ds_name else 0
      # Also, with no preprocess_fn specified (i.e., default), the images for
      # the in- and out-of-distribution datasets can have different shapes.
      if batch_index * batch_size <= NUM_EXAMPLES:
        in_labels = [in_label] * batch_size
        self.assertAllEqual(features["label"], in_labels)
        self.assertAllEqual(features["metadata"]["label"], in_labels)
        self.assertEqual(features["image"].shape, in_ds_expected_shape)
      else:
        ood_labels = [ood_label] * batch_size
        self.assertAllEqual(features["label"], ood_labels)
        self.assertAllEqual(features["metadata"]["label"], ood_labels)
        self.assertEqual(features["image"].shape, out_ds_expected_shape)

  @parameterized.parameters([
      # cifar10 versus *
      ("cifar10_vs_cifar100",),
      ("cifar10_vs_dtd",),
      ("cifar10_vs_places365",),
      ("cifar10_vs_svhn",),
      ("cifar10_vs_cifar100(ood_with_positive_labels=True)",),
      ("cifar10_vs_dtd(ood_with_positive_labels=True)",),
      ("cifar10_vs_places365(ood_with_positive_labels=True)",),
      ("cifar10_vs_svhn(ood_with_positive_labels=True)",),
      # cifar100 versus *
      ("cifar100_vs_cifar10",),
      ("cifar100_vs_dtd",),
      ("cifar100_vs_places365",),
      ("cifar100_vs_svhn",),
      ("cifar100_vs_cifar10(ood_with_positive_labels=True)",),
      ("cifar100_vs_dtd(ood_with_positive_labels=True)",),
      ("cifar100_vs_places365(ood_with_positive_labels=True)",),
      ("cifar100_vs_svhn(ood_with_positive_labels=True)",),
      # imagenet versus *
      ("imagenet_vs_places365",),
      ("imagenet_vs_places365(ood_with_positive_labels=True)",),
  ])
  def test_that_it_preprocesses_and_batches(self, name, batch_size=8):

    def preprocess_fn(features):
      features["image"] = tf.image.resize(features["image"], [224, 224])
      features["foo"] = tf.constant(1, dtype=tf.int64)
      return features

    dataset = rm.datasets.get(name).load(
        preprocess_fn=preprocess_fn).batch(batch_size)

    for features in dataset.take(1):
      for feature in self.necessary_fields + [
          "label", "foo"
      ]:
        self.assertIn(feature, features.keys())
      self.assertAllEqual(features["foo"], [1] * batch_size)
      self.assertAllEqual(features["image"].shape, [batch_size, 224, 224, 3])

  @parameterized.parameters([
      # cifar10 versus *
      ("cifar10_vs_cifar100", C10, C100,
       ["id", "image", "label", "metadata"], ["element_id", "label"]),
      ("cifar10_vs_dtd", C10, rm_tfds.DtdDataset,
       ["image", "label", "metadata"], ["element_id", "label"]),
      ("cifar10_vs_places365", C10, rm_tfds.Places365Dataset,
       ["image", "label", "metadata"], ["element_id", "label"]),
      ("cifar10_vs_svhn", C10, rm_tfds.SvhnDataset,
       ["image", "label", "metadata"], ["element_id", "label"]),
      # cifar100 versus *
      ("cifar100_vs_cifar10", C100, C10,
       ["id", "image", "label", "metadata"], ["element_id", "label"]),
      ("cifar100_vs_dtd", C100, rm_tfds.DtdDataset,
       ["image", "label", "metadata"], ["element_id", "label"]),
      ("cifar100_vs_places365", C100, rm_tfds.Places365Dataset,
       ["image", "label", "metadata"], ["element_id", "label"]),
      ("cifar100_vs_svhn", C100, rm_tfds.SvhnDataset,
       ["image", "label", "metadata"], ["element_id", "label"]),
      # imagenet versus *
      ("imagenet_vs_places365", ImageNet, rm_tfds.Places365Dataset,
       ["image", "label", "metadata"], ["element_id", "label"]),
  ])
  def test_common_feature_keys(self, ds_name, in_ds_type, out_ds_type,
                               expected_keys, expected_metadata_keys):
    dataset = rm.datasets.get(ds_name)
    self.assertIsInstance(dataset.in_dataset, in_ds_type)
    self.assertIsInstance(dataset.out_dataset, out_ds_type)
    self.assertEqual(dataset.info.num_classes, 2)

    dataset = dataset.load(preprocess_fn=None)
    self.assertSameElements(dataset.element_spec.keys(), expected_keys)
    self.assertSameElements(dataset.element_spec["metadata"].keys(),
                            expected_metadata_keys)


if __name__ == "__main__":
  tf.compat.v1.enable_eager_execution()
  with tfds.testing.mock_data(num_examples=NUM_EXAMPLES):
    tf.test.main()
