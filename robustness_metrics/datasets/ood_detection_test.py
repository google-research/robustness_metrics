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
"""Tests for OOD detection datasets."""

from absl.testing import parameterized

import robustness_metrics as rm
from robustness_metrics.datasets import tfds as rm_tfds
import tensorflow as tf
import tensorflow_datasets as tfds


NUM_EXAMPLES = 32


class OodDetectionDatasetsTest(parameterized.TestCase, tf.test.TestCase):

  necessary_fields = ["image", "metadata"]

  @parameterized.parameters([
      ("cifar10_vs_cifar100",),
      ("cifar100_vs_cifar10",),
  ])
  def test_that_it_loads_with_default(self, ds_name, label_field="label"):
    expected_shape = [32, 32, 3]
    batch_size = 8
    assert_msg = ("The unit test requires NUM_EXAMPLES to be a multiple of the "
                  f" batch_size; received {NUM_EXAMPLES} and {batch_size}.")
    assert NUM_EXAMPLES % batch_size == 0, assert_msg

    # The in- and out-of-distribution datasets are concatenated, with a total
    # size equal to 2 * NUM_EXAMPLES.
    dataset = rm.datasets.get(ds_name)
    dataset = dataset.load(preprocess_fn=None)
    self.assertEqual(dataset.cardinality(), 2 * NUM_EXAMPLES)

    dataset = dataset.batch(batch_size)
    for batch_index, features in enumerate(dataset, 1):
      for feature in self.necessary_fields + [label_field]:
        self.assertIn(feature, features.keys())
      self.assertEqual(features["image"].shape, [batch_size] + expected_shape)
      self.assertEqual(features["metadata"]["element_id"].dtype, tf.int64)
      # The first NUM_EXAMPLES datapoints are the in-distribution datapoints.
      # Their labels are 1's. Conversely, the last NUM_EXAMPLES datapoints are
      # the out-of-distribution datapoints, with labels equal to 0's.
      if batch_index * batch_size <= NUM_EXAMPLES:
        self.assertAllEqual(features["label"], [1] * batch_size)
      else:
        self.assertAllEqual(features["label"], [0] * batch_size)

  @parameterized.parameters([
      ("cifar10_vs_cifar100",),
      ("cifar100_vs_cifar10",),
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
      ("cifar10_vs_cifar100", rm_tfds.Cifar10Dataset, rm_tfds.Cifar100Dataset,
       ["id", "image", "label", "metadata"], ["element_id", "label"]),
      ("cifar100_vs_cifar10", rm_tfds.Cifar100Dataset, rm_tfds.Cifar10Dataset,
       ["id", "image", "label", "metadata"], ["element_id", "label"]),
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
