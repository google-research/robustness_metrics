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

"""Tests for robustness_metrics OOD detection reports."""
from absl.testing import absltest
import robustness_metrics as rm


def _add_arg_to_metric_name(metric_name, dataset_name):
  # For OOD detection, there are two conventions:
  #  (i) The OOD dataset has the negative labels, and we use MSP (i.e., "maximum
  #      softmax probability") to go from multiclass to binary detection.
  #  (ii) Conversely, the OOD dataset has the positive labels so we use 1 - MSP.
  # See more background in /robustness_metrics/metrics/retrieval.py.
  arg = "ood_with_positive_labels=True" in dataset_name
  return f"{metric_name}(one_minus_msp={arg})"


class OodDetectionReportTest(absltest.TestCase):

  def test_cifar10_report(self):

    report = rm.reports.ood_detection.Cifar10OodDetectionReport()
    metric_names = ["auc_pr", "auc_roc", "fpr95"]
    ood_datasets = [
        f"{dataset}(ood_with_positive_labels={arg})"   # pylint: disable=g-complex-comprehension
        for dataset in ["cifar100", "dtd", "places365", "svhn"]
        for arg in [True, False]
    ]
    for d in ood_datasets:
      for mocked_metric_value, metric_name in enumerate(metric_names):
        metric_name_with_arg = _add_arg_to_metric_name(metric_name, d)
        report.add_measurement(f"cifar10_vs_{d}", metric_name_with_arg,
                               {metric_name: mocked_metric_value})

    expected_results = {  # pylint: disable=g-complex-comprehension
        f"cifar10_vs_{d}/{_add_arg_to_metric_name(m, d)}/{m}": v
        for d in ood_datasets for v, m in enumerate(metric_names)
    }
    self.assertEqual(report.result(), expected_results)

  def test_cifar100_report(self):

    report = rm.reports.ood_detection.Cifar100OodDetectionReport()
    metric_names = ["auc_pr", "auc_roc", "fpr95"]
    ood_datasets = [
        f"{dataset}(ood_with_positive_labels={arg})"   # pylint: disable=g-complex-comprehension
        for dataset in ["cifar10", "dtd", "places365", "svhn"]
        for arg in [True, False]
    ]
    for d in ood_datasets:
      for mocked_metric_value, metric_name in enumerate(metric_names):
        metric_name_with_arg = _add_arg_to_metric_name(metric_name, d)
        report.add_measurement(f"cifar100_vs_{d}", metric_name_with_arg,
                               {metric_name: mocked_metric_value})

    expected_results = {  # pylint: disable=g-complex-comprehension
        f"cifar100_vs_{d}/{_add_arg_to_metric_name(m, d)}/{m}": v
        for d in ood_datasets for v, m in enumerate(metric_names)
    }
    self.assertEqual(report.result(), expected_results)

  def test_imagenet_report(self):

    report = rm.reports.ood_detection.ImagenetOodDetectionReport()
    metric_names = ["auc_pr", "auc_roc", "fpr95"]
    ood_datasets = [
        f"places365(ood_with_positive_labels={arg})"
        for arg in [True, False]
    ]
    for d in ood_datasets:
      for mocked_metric_value, metric_name in enumerate(metric_names):
        metric_name_with_arg = _add_arg_to_metric_name(metric_name, d)
        report.add_measurement(f"imagenet_vs_{d}", metric_name_with_arg,
                               {metric_name: mocked_metric_value})

    expected_results = {  # pylint: disable=g-complex-comprehension
        f"imagenet_vs_{d}/{_add_arg_to_metric_name(m, d)}/{m}": v
        for d in ood_datasets for v, m in enumerate(metric_names)
    }
    self.assertEqual(report.result(), expected_results)

if __name__ == "__main__":
  absltest.main()
