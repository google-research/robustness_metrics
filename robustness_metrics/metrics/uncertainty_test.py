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
"""Tests for uncertainty metrics."""

import math
from absl.testing import parameterized
import numpy as np
import robustness_metrics as rm
import tensorflow as tf

_UNCERTAINTY_METRICS = [
    "ece", "nll", "brier",
    "adaptive_ece(datapoints_per_bin=1,threshold=0.0)",
    "adaptive_ece(datapoints_per_bin=1,threshold=0.0,temperature_scaling=True)",
]


def _get_info(num_classes=2):
  return rm.datasets.base.DatasetInfo(num_classes=num_classes)


class KerasMetricTest(parameterized.TestCase, tf.test.TestCase):

  def assertDictsAlmostEqual(self, dict_1, dict_2):
    self.assertEqual(dict_1.keys(), dict_2.keys())
    for key in dict_1:
      self.assertAlmostEqual(dict_1[key], dict_2[key], places=5)

  @parameterized.parameters([(name,) for name in _UNCERTAINTY_METRICS])
  def test_binary_prediction_two_predictions_per_element(self, name):
    metric = rm.metrics.get(name, _get_info(2))
    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=1,
            metadata={"label": 1},
            predictions=[[.2, .8], [.7, .3]]))
    expected_value = {
        # Checked manually:
        "adaptive_ece(datapoints_per_bin=1,threshold=0.0)": .45,
        "adaptive_ece(datapoints_per_bin=1,threshold=0.0,temperature_scaling=True)":
            0.0,
        "ece": .45,
        "nll": -math.log((.8 + .3) / 2),
        "brier": ((.2 + .7) / 2)**2 + ((.2 + .7) / 2)**2,
    }[name]
    key_name = name.split("(")[0]
    self.assertDictsAlmostEqual(metric.result(), {key_name: expected_value})

  @parameterized.parameters([(name,) for name in _UNCERTAINTY_METRICS])
  def test_binary_predictions_on_different_predictions(self, name):
    metric = rm.metrics.get(name, _get_info(2))
    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=1, metadata={"label": 1}, predictions=[[.2, .8]]))
    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=2, metadata={"label": 0}, predictions=[[.3, .7]]))
    expected_value = {
        # Checked manually:
        "adaptive_ece(datapoints_per_bin=1,threshold=0.0)": .45,
        "adaptive_ece(datapoints_per_bin=1,threshold=0.0,temperature_scaling=True)":
            0.4733536,  # TODO(mjlm) check the math manually!
        "ece": .45,
        "nll": 0.5 * (-math.log(.8) - math.log(.3)),
        "brier": 0.5 * (.2**2 + .2**2 + .7**2 + .7**2),
    }[name]
    key_name = name.split("(")[0]
    self.assertDictsAlmostEqual(metric.result(), {key_name: expected_value})

  @parameterized.parameters([(name,) for name in _UNCERTAINTY_METRICS])
  def test_tertiary_prediction(self, name):
    metric = rm.metrics.get(name, _get_info(3))
    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=1,
            metadata={"label": 2},
            predictions=[[.2, .4, .4], [.5, .3, .2]]))
    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=2, metadata={"label": 1}, predictions=[[.8, .15, .05]]))
    expected_value = {
        # Checked manually:
        "adaptive_ece(datapoints_per_bin=1,threshold=0.0)":
            .51666666,
        "adaptive_ece(datapoints_per_bin=1,threshold=0.0,temperature_scaling=True)":
            0.4405967079878724,  # TODO(mjlm) check the math manually!
        "ece":
            0.575,
        "nll":
            -0.5 * (math.log((.4 + .2) / 2) + math.log(.15)),
        "brier":
            0.5 * (((.2 + .5) / 2)**2 + ((.4 + .3) / 2)**2 +
                   ((.6 + .8) / 2)**2 + .8**2 + .85**2 + .05**2)
    }[name]
    key_name = name.split("(")[0]
    self.assertDictsAlmostEqual(metric.result(), {key_name: expected_value})

  def test_scale_predictions(self):
    predictions = [[0.10, 0.17, 0.18, 0.11, 0.44],
                   [0.10, 0.10, 0.20, 0.20, 0.40],
                   [0.18, 0.26, 0.14, 0.03, 0.39],
                   [0.52, 0.04, 0.10, 0.32, 0.02]]
    labels = np.argmax(predictions, axis=-1)
    labels[0:2] = 0  # Add some noise.
    true_beta = 0.581074  # Computed manually
    scaled_preds = rm.metrics.uncertainty._scale_predictions(
        predictions, labels)
    np.testing.assert_almost_equal(
        scaled_preds,
        tf.nn.softmax(true_beta * np.log(predictions)).numpy(),
        decimal=5)


if __name__ == "__main__":
  tf.test.main()
