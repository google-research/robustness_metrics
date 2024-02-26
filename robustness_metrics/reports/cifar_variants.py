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

  def _yield_classification_specs(self, dataset):
    for metric in ["accuracy", "ece", "nll", "brier", "timing"]:
      yield base.MeasurementSpec(dataset, metric)

  @property
  def required_measurements(self):
    yield from self._yield_classification_specs("cifar10")

    severities = list(range(1, 6))
    for corruption_type in CIFAR10_CORRUPTIONS:
      for severity in severities:
        dataset = (f"cifar10_c(corruption_type={corruption_type!r},"
                   f"severity={severity})")
        yield from self._yield_classification_specs(dataset)

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


@base.registry.register("cifar10_variants_ensemble")
class Cifar10VariantsEnsembleReport(Cifar10VariantsReport):
  """Aggregated statistics over the CIFAR-10 variants in the case of ensembles.

  It contains the same metrics as in `cifar10_variants`, with the addition of
  the diversity metrics.
  """

  def _yield_classification_specs(self, dataset):
    yield from super()._yield_classification_specs(dataset)
    yield base.MeasurementSpec(
        dataset, "average_pairwise_diversity(normalize_disagreement=True)")
    yield base.MeasurementSpec(
        dataset, "average_pairwise_diversity(normalize_disagreement=False)")

  def _get_full_metric_key(self, dataset_name, metric_name, metric_key):
    _, _, diversity_metric_kwargs = registry.parse_name_and_kwargs(metric_name)
    is_normalized = diversity_metric_kwargs["normalize_disagreement"]
    if metric_key == "disagreement" and is_normalized:
      full_metric_key = f"{dataset_name}/normalized_{metric_key}"
    else:
      full_metric_key = f"{dataset_name}/{metric_key}"
    return full_metric_key

  def add_measurement(self, dataset_spec, metric_name, metric_results):
    if metric_name not in [
        "average_pairwise_diversity(normalize_disagreement=True)",
        "average_pairwise_diversity(normalize_disagreement=False)"
    ]:
      super().add_measurement(dataset_spec, metric_name, metric_results)
    else:
      dataset_name, _, _ = registry.parse_name_and_kwargs(dataset_spec)
      for metric_key, metric_value in metric_results.items():
        key = self._get_full_metric_key(dataset_name, metric_name, metric_key)
        if dataset_name == "cifar10_c":
          self._corruption_metrics[key].append(metric_value)
        else:
          self._results[key] = metric_value
