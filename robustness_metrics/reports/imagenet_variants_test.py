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

"""Tests for robustness_metrics.reports.imagenet_variants."""
from absl.testing import absltest
from absl.testing import parameterized
import robustness_metrics as rm


class ImagenetVariantsTest(parameterized.TestCase):

  @parameterized.parameters(
      ("dataset1", "average_pairwise_diversity(normalize_disagreement=False)",
       "great_metric", "dataset1/great_metric"),
      ("dataset1", "average_pairwise_diversity(normalize_disagreement=True)",
       "great_metric", "dataset1/great_metric"),
      ("dataset2", "average_pairwise_diversity(normalize_disagreement=False)",
       "disagreement", "dataset2/disagreement"),
      ("dataset3", "average_pairwise_diversity(normalize_disagreement=True)",
       "disagreement", "dataset3/normalized_disagreement"))
  def test_get_full_metric_key(self, dataset_name, metric_name, metric_key,
                               expected_key):

    r = rm.reports.imagenet_variants.ImagenetVariantsEnsembleGceSweepReport()
    key = r._get_full_metric_key(dataset_name, metric_name, metric_key)

    self.assertEqual(key, expected_key)

  def test_add_measurement(self):

    r = rm.reports.imagenet_variants.ImagenetVariantsEnsembleGceSweepReport()

    # first add.
    dataset_spec = "imagenet_c(corruption_type='contrast',severity=4)"
    metric_name = "average_pairwise_diversity(normalize_disagreement=True)"
    metric_results = {"metric1": 1.0, "disagreement": -1.0}
    r.add_measurement(dataset_spec, metric_name, metric_results)

    self.assertEmpty(r._results)
    self.assertEmpty(r._accuracy_per_corruption)

    expected_corruption_metrics = {
        "imagenet_c/metric1": [1.0],
        "imagenet_c/normalized_disagreement": [-1.0],
    }
    self.assertEqual(r._corruption_metrics, expected_corruption_metrics)

    # Second add.
    dataset_spec = "imagenet_c(corruption_type='contrast',severity=2)"
    metric_name = "accuracy"
    metric_results = {"accuracy": 0.1}
    r.add_measurement(dataset_spec, metric_name, metric_results)

    self.assertEmpty(r._results)

    expected_corruption_metrics = {
        "imagenet_c/metric1": [1.0],
        "imagenet_c/normalized_disagreement": [-1.0],
        "imagenet_c/accuracy": [0.1]
    }
    self.assertEqual(r._corruption_metrics, expected_corruption_metrics)

    expected_accuracy_per_corruption = {"contrast": [0.1]}
    self.assertEqual(r._accuracy_per_corruption,
                     expected_accuracy_per_corruption)

    # Third add.
    dataset_spec = "imagenet"
    metric_name = "average_pairwise_diversity(normalize_disagreement=False)"
    metric_results = {"metric2": 10.0, "disagreement": -3.0}
    r.add_measurement(dataset_spec, metric_name, metric_results)

    expected_results = {
        "imagenet/metric2": 10.0,
        "imagenet/disagreement": -3.0
    }
    self.assertEqual(r._results, expected_results)
    self.assertEqual(r._corruption_metrics, expected_corruption_metrics)
    self.assertEqual(r._accuracy_per_corruption,
                     expected_accuracy_per_corruption)


if __name__ == "__main__":
  absltest.main()
