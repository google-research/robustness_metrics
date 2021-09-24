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
"""Tests for synthetic metrics."""

from absl.testing import parameterized
import numpy as np
import robustness_metrics as rm
import tensorflow as tf


class SyntheticTest(parameterized.TestCase, tf.test.TestCase):

  def assertDictsAlmostEqual(self, dict_1, dict_2):
    self.assertEqual(dict_1.keys(), dict_2.keys())
    for key in dict_1:
      self.assertAlmostEqual(dict_1[key], dict_2[key])

  def test_mapping(self):

    metric = rm.metrics.base.get("synthetic")(dataset_info=None)

    data = [
        ("size", 0, "size(0.01)",),
        ("size", 101022, "size(0.83)",),
        ("location", 10036, "location(0.00,0.40)",),
        ("location", 1000, "location(0.00,0.00)",),
        ("rotation", 10018, "rotation(161)",),
        ("rotation", 10942, "rotation(161)"),
    ]
    for dataset_variant, image_id, expected_group in data:
      group = metric.map_path_to_group(image_id, dataset_variant)
      self.assertEqual(group, expected_group)

  def test_grouping(self):

    def _one_hot(x):
      ret = np.zeros(1000)
      ret[x] = 1
      return ret

    metric = rm.metrics.base.get("synthetic")(dataset_info=None)

    # Area-related instances
    metric.add_predictions(
        _one_hot(400),
        metadata={
            "image_id": 0,
            "dataset_variant": "size",
            "label": 407,
            "element_id": 1,
        })

    metric.add_predictions(
        _one_hot(407),
        metadata={
            "image_id": 101022,
            "dataset_variant": "size",
            "label": 407,
            "element_id": 2,
        })

    # Location-related instances
    metric.add_predictions(
        _one_hot(100),
        metadata={
            "image_id": 10036,
            "dataset_variant": "location",
            "label": 529,
            "element_id": 3,
        })

    metric.add_predictions(
        _one_hot(806),
        metadata={
            "image_id": 1000,
            "dataset_variant": "location",
            "label": 806,
            "element_id": 4,
        })

    # Rotation instance
    metric.add_predictions(
        _one_hot(806),
        metadata={
            "image_id": 10942,
            "dataset_variant": "rotation",
            "label": 806,
            "element_id": 5,
        })

    self.assertDictsAlmostEqual(
        metric.result(),
        {
            "size(0.01)": 0.0,  # 0/1 correct
            "size(0.83)": 1.0,  # 1/1 correct
            "location(0.00,0.00)": 1.0,  # 1/1 correct
            "location(0.00,0.40)": 0.0,  # 0/1 correct
            "rotation(161)": 1.0,  # 2/2 correct
            "size_average": 0.5,  # Two groups, avg([0.0, 1.0]) = 0.5
            "location_average": 0.5,  # Two groups, avg([0.5, 1.0]) = 0.5
            "rotation_average": 1.0,  # One group, avg([1.0]) = 1.0
        })


if __name__ == "__main__":
  tf.test.main()
