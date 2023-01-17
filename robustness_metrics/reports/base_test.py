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

"""Tests for robustness_metrics.reports."""
from absl.testing import absltest
import robustness_metrics as rm


class UnionReportTest(absltest.TestCase):

  def test_with_empty_required_measurements(self):

    class EmptyReport(rm.reports.base.UnionReport):
      required_measurements = []

    report = EmptyReport()
    self.assertEqual(report.result(), {})

  def test_with_correct_arguments(self):

    class TestReport(rm.reports.base.UnionReport):
      required_measurements = [
          rm.reports.base.MeasurementSpec("dataset-1", "metric-1"),
          rm.reports.base.MeasurementSpec("dataset-1", "metric-2"),
          rm.reports.base.MeasurementSpec("dataset-2", "metric-1")
      ]

    report = TestReport()
    report.add_measurement("dataset-1", "metric-1", {"a": 1, "b": 2})
    report.add_measurement("dataset-2", "metric-1", {"a": 3, "b": 4})
    report.add_measurement("dataset-1", "metric-2", {"c": 5, "d": 6})
    self.assertEqual(
        report.result(), {
            "dataset-1/metric-1/a": 1,
            "dataset-1/metric-1/b": 2,
            "dataset-2/metric-1/a": 3,
            "dataset-2/metric-1/b": 4,
            "dataset-1/metric-2/c": 5,
            "dataset-1/metric-2/d": 6,
        })

  def test_that_exception_is_raised_on_too_few_observations(self):

    class TestReport(rm.reports.base.UnionReport):
      required_measurements = [
          rm.reports.base.MeasurementSpec("dataset-1", "metric-1"),
          rm.reports.base.MeasurementSpec("dataset-1", "metric-2"),
          rm.reports.base.MeasurementSpec("dataset-2", "metric-1")
      ]

    report = TestReport()
    report.add_measurement("dataset-1", "metric-1", {"a": 1, "b": 2})
    report.add_measurement("dataset-1", "metric-2", {"c": 5, "d": 6})
    with self.assertRaises(ValueError):
      report.result()

  def test_that_exception_is_raised_on_unknown_observations(self):

    class TestReport(rm.reports.base.UnionReport):
      required_measurements = [
          rm.reports.base.MeasurementSpec("dataset-1", "metric-1"),
          rm.reports.base.MeasurementSpec("dataset-1", "metric-2"),
          rm.reports.base.MeasurementSpec("dataset-2", "metric-1")
      ]

    report = TestReport()
    report.add_measurement("dataset-1", "metric-1", {"a": 1, "b": 2})
    report.add_measurement("dataset-1", "metric-2", {"c": 5, "d": 6})
    with self.assertRaises(ValueError):
      report.add_measurement("dataset-1", "metric-3", {"c": 5, "d": 6})


class EnsembleClassificationReportTest(absltest.TestCase):

  def test_required_measurements(self):

    report = rm.reports.base.EnsembleClassficationReport(["cifar10"])
    metric_names = ["accuracy", "nll", "ece", "brier"]
    for mocked_metric_value, metric_name in enumerate(metric_names):
      report.add_measurement("cifar10", metric_name,
                             {metric_name: mocked_metric_value})

    # Diversity metrics are missing, an error is expected.
    with self.assertRaises(ValueError):
      report.result()

    div_metric_names = [
        "average_pairwise_diversity(normalize_disagreement=True)",
        "average_pairwise_diversity(normalize_disagreement=False)"
    ]
    for mocked_metric_value, metric_name in enumerate(div_metric_names):
      report.add_measurement("cifar10", metric_name,
                             {metric_name: mocked_metric_value})

    self.assertEqual(
        report.result(), {
            "cifar10/accuracy/accuracy": 0,
            "cifar10/nll/nll": 1,
            "cifar10/ece/ece": 2,
            "cifar10/brier/brier": 3,
            f"cifar10/{div_metric_names[0]}/{div_metric_names[0]}": 0,
            f"cifar10/{div_metric_names[1]}/{div_metric_names[1]}": 1,
        })

if __name__ == "__main__":
  absltest.main()
