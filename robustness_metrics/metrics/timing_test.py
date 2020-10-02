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
"""Tests for timing metrics."""

from absl.testing import parameterized
import numpy as np
import robustness_metrics as rm
import tensorflow as tf


def _get_info(num_classes=2):
  return rm.datasets.base.DatasetInfo(num_classes=num_classes)


class KerasMetricTest(parameterized.TestCase, tf.test.TestCase):

  def test_on_example_input(self):
    metric = rm.metrics.get("timing", _get_info(2))
    timings = [
        0.47742076, 0.40866531, 0.04159123, 0.53712302, 0.25981041, 0.71751233,
        0.14699851, 0.85301045, 0.04667912, 0.84366503, 0.27481068, 0.55640206
    ]
    for timing in timings:
      metric.add_predictions(rm.common.types.ModelPredictions(
          time_in_s=timing, element_id=None, predictions=[], metadata=None))

    for key, value in metric.result().items():
      if key == "mean":
        self.assertAlmostEqual(value, np.mean(timings))
      else:
        assert key.startswith("quantile/")
        quantile = float(key[9:])
        self.assertEqual(value, np.quantile(timings, quantile))


if __name__ == "__main__":
  tf.test.main()
