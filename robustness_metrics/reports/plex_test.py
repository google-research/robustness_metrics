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

"""Tests for robustness_metrics.reports.plex."""
from absl.testing import absltest
from absl.testing import parameterized
import robustness_metrics as rm


class PlexTest(parameterized.TestCase):

  def test_add_measurement(self):

    r = rm.reports.plex.ImagenetPlexReport(convert_imagenet_real_labels=False)

    # first add.
    dataset_spec = "imagenet_c(corruption_type='contrast',severity=4)"
    metric_name = "collaborative_auc"
    metric_results = {"metric0": 5.0, "metric1": -1.0}
    r.add_measurement(dataset_spec, metric_name, metric_results)

    self.assertEmpty(r._results)
    self.assertEmpty(r._accuracy_per_corruption)

    expected_corruption_metrics = {
        "imagenet_c/metric0": [5.0],
        "imagenet_c/metric1": [-1.0],
    }
    self.assertEqual(r._corruption_metrics, expected_corruption_metrics)

    # Second add.
    dataset_spec = "imagenet_c(corruption_type='contrast',severity=2)"
    metric_name = "accuracy"
    metric_results = {"metric2": 0.1}
    r.add_measurement(dataset_spec, metric_name, metric_results)

    self.assertEmpty(r._results)

    expected_corruption_metrics = {
        "imagenet_c/metric0": [5.0],
        "imagenet_c/metric1": [-1.0],
        "imagenet_c/accuracy": [0.1]
    }
    self.assertEqual(r._corruption_metrics, expected_corruption_metrics)

    expected_accuracy_per_corruption = {"contrast": [0.1]}
    self.assertEqual(r._accuracy_per_corruption,
                     expected_accuracy_per_corruption)

    # Third add.
    dataset_spec = "imagenet_real(convert_labels=True)"
    metric_name = "nll(soft_labels=True)"
    metric_results = {"metric2": -3.0}
    r.add_measurement(dataset_spec, metric_name, metric_results)

    dataset_spec = "imagenet_vs_places365(ood_with_positive_labels=True)"
    metric_name = "fpr95(one_minus_msp=True)"
    metric_results = {"metric3": 12.0}
    r.add_measurement(dataset_spec, metric_name, metric_results)

    expected_results = {
        "imagenet_real/metric2": -3.0,
        "imagenet_vs_places365/metric3": 12.0
    }
    self.assertEqual(r._results, expected_results)
    self.assertEqual(r._corruption_metrics, expected_corruption_metrics)
    self.assertEqual(r._accuracy_per_corruption,
                     expected_accuracy_per_corruption)

  def test_required_measurements(self):
    r = rm.reports.plex.ImagenetPlexReport(convert_imagenet_real_labels=False)
    self.assertLen(list(r.required_measurements), 643)


if __name__ == "__main__":
  absltest.main()
