# coding=utf-8
# Copyright 2022 The Robustness Metrics Authors.
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
"""Tests for robustness_metrics OOD detection reports."""
from absl.testing import absltest
import robustness_metrics as rm


class OodDetectionReportTest(absltest.TestCase):

  def test_cifar10_report(self):

    report = rm.reports.ood_detection.Cifar10OodDetectionReport()
    metric_names = ["auc_pr", "auc_roc", "fpr95"]
    ood_datasets = ["cifar100", "dtd", "places365", "svhn"]
    for d in ood_datasets:
      for mocked_metric_value, metric_name in enumerate(metric_names):
        report.add_measurement(f"cifar10_vs_{d}", metric_name,
                               {metric_name: mocked_metric_value})

    expected_results = {   # pylint: disable=g-complex-comprehension
        f"cifar10_vs_{d}/{m}/{m}": v for d in ood_datasets
        for v, m in enumerate(metric_names)
    }
    self.assertEqual(report.result(), expected_results)

  def test_cifar100_report(self):

    report = rm.reports.ood_detection.Cifar100OodDetectionReport()
    metric_names = ["auc_pr", "auc_roc", "fpr95"]
    ood_datasets = ["cifar10", "dtd", "places365", "svhn"]
    for d in ood_datasets:
      for mocked_metric_value, metric_name in enumerate(metric_names):
        report.add_measurement(f"cifar100_vs_{d}", metric_name,
                               {metric_name: mocked_metric_value})

    expected_results = {   # pylint: disable=g-complex-comprehension
        f"cifar100_vs_{d}/{m}/{m}": v for d in ood_datasets
        for v, m in enumerate(metric_names)
    }
    self.assertEqual(report.result(), expected_results)

if __name__ == "__main__":
  absltest.main()
