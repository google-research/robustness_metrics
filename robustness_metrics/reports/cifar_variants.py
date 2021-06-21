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
"""Reports that evaluate models over the CIFAR-10 and CIFAR-100 label space."""
import collections
from typing import Dict

from robustness_metrics.common import registry
from robustness_metrics.reports import base

CIFAR10_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "frosted_glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic",
    "pixelate",
    "jpeg_compression",
]


@base.registry.register("cifar10_variants")
class Cifar10VariantsReport(base.Report):
  """Aggregated statistics over the CIFAR-10 variants.

  This report contains the following CIFAR-10 variants: cifar10,
  cifar10_c (all variants). For each dataset, we compute accuracy,
  expected calibration error (ece), log-likelihood, Brier, timing, and
  adaptive ECE.
  """

  def __init__(self):
    super().__init__()
    self._corruption_metrics = collections.defaultdict(list)
    self._results = {}

  @property
  def required_measurements(self):
    def _yield_classification_specs(dataset):
      for metric in ["accuracy", "ece", "nll", "brier", "timing"]:
        yield base.MeasurementSpec(dataset, metric)

    yield from _yield_classification_specs("cifar10")

    severities = list(range(1, 6))
    for corruption_type in CIFAR10_CORRUPTIONS:
      for severity in severities:
        dataset = (f"cifar10_c(corruption_type={corruption_type!r},"
                   f"severity={severity})")
        yield from _yield_classification_specs(dataset)

  def add_measurement(self, dataset_spec, metric_name, metric_results):
    dataset_name, _, _ = registry.parse_name_and_kwargs(dataset_spec)
    if dataset_name == "cifar10_c":
      if metric_name == "timing":
        value = metric_results["mean"]
      else:
        # All remaining metrics are a dictionary with a single value, with
        # the key equal to the metric name.
        value, = list(metric_results.values())
      self._corruption_metrics[f"cifar10_c/{metric_name}"].append(value)
    else:
      if metric_name == "timing":
        value = metric_results["mean"]
      else:
        # All remaining metrics are a dictionary with a single value, with
        # the key equal to the metric name.
        value, = list(metric_results.values())
      self._results[f"{dataset_name}/{metric_name}"] = value

  def result(self) -> Dict[str, float]:
    results = dict(self._results)
    results.update(base.compute_stats_per_bucket(self._corruption_metrics))
    return results
