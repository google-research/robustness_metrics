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

"""Tests for datasets from TFDS."""

from absl.testing import parameterized

import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds


class TaskAdaptationDatasetsTest(parameterized.TestCase, tf.test.TestCase):

  necessary_fields = ["image", "metadata"]

  @parameterized.parameters([
      ("cifar10",),
      ("cifar100",),
      ("cifar10_c(corruption_type='gaussian_noise',severity=1)"),
      ("imagenet",),
      ("imagenet_a",),
      ("imagenet_c(corruption_type='gaussian_noise',severity=1)",),
      ("imagenet_v2(variant='MATCHED_FREQUENCY')",),
      ("synthetic(variant='location')",),
      ("oxford_flowers102",),
      ("oxford_iiit_pet",),
      ("places365",),
      ("dtd",),
      ("svhn",),
      ("imagenet_sketch",),
  ])
  def test_that_it_loads_with_default(self, name, label_field="label"):
    dataset_object = dataset = rm.datasets.get(name)
    try:
      num_classes = {
          "cifar10": 10,
          "cifar100": 100,
          "cifar10_c(corruption_type='gaussian_noise',severity=1)": 10,
          "oxford_flowers102": 102,
          "oxford_iiit_pet": 37,
          "places365": 365,
          "svhn": 10,
          "dtd": 47,
      }[name]
    except KeyError:
      num_classes = 1000
    self.assertEqual(dataset_object.info.num_classes, num_classes)
    if "cifar" in name:
      expected_shape = [32, 32, 3]
    else:
      expected_shape = [224, 224, 3]
    dataset = dataset_object.load(preprocess_fn=None).batch(8)
    for features in dataset.take(1):
      for feature in self.necessary_fields + [label_field]:
        self.assertIn(feature, features.keys())
      self.assertEqual(features["image"].shape, [8] + expected_shape)
      self.assertEqual(features["metadata"]["element_id"].dtype, tf.int64)

  def test_that_it_preprocesses_and_batches(self, batch_size=8):

    def preprocess_fn(features):
      features["image"] = tf.image.resize(features["image"], [224, 224])
      features["foo"] = tf.constant(1, dtype=tf.int64)
      return features

    dataset = rm.datasets.get("imagenet").load(
        preprocess_fn=preprocess_fn).batch(batch_size)

    for features in dataset.take(1):
      for feature in self.necessary_fields + [
          "label", "foo"
      ]:
        self.assertIn(feature, features.keys())
      self.assertAllEqual(features["foo"], [1] * batch_size)


if __name__ == "__main__":
  tf.compat.v1.enable_eager_execution()
  with tfds.testing.mock_data(num_examples=32):
    tf.test.main()
