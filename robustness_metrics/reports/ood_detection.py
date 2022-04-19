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

"""Reports about OOD detection for {cifar10, cifar100} versus other datasets."""
from typing import List
from robustness_metrics.reports import base


class OodDetectionReport(base.UnionReport):
  """Computes commonly used metrics for OOD detection tasks.

  See https://arxiv.org/pdf/2106.03004.pdf for more background.
  """

  def __init__(self, datasets_specs: List[str]):
    self._datasets = datasets_specs
    super().__init__()

  @property
  def required_measurements(self):
    metrics = ["auc_pr", "auc_roc", "fpr95"]
    for dataset in self._datasets:
      for metric in metrics:
        yield base.MeasurementSpec(dataset, metric)


@base.registry.register("cifar10_ood_detection_report")
class Cifar10OodDetectionReport(OodDetectionReport):

  def __init__(self):
    ood_datasets = ["cifar100", "dtd", "places365", "svhn"]
    super().__init__(datasets_specs=[f"cifar10_vs_{d}" for d in ood_datasets])


@base.registry.register("cifar100_ood_detection_report")
class Cifar100OodDetectionReport(OodDetectionReport):

  def __init__(self):
    ood_datasets = ["cifar10", "dtd", "places365", "svhn"]
    super().__init__(datasets_specs=[f"cifar100_vs_{d}" for d in ood_datasets])
