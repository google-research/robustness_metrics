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
"""Reports that evaluate models over the ImageNet label space."""
import collections
from typing import Dict
from absl import logging
import numpy as np
from robustness_metrics.common import registry
from robustness_metrics.reports import base


# The numbers are from Appendix C in
#   "Benchmarking Neural Network Robustness to Common Corruptions and Surface
#    Variations", Hendrycks and Diettrich, ICLR 2019
ALEXENET_AVERAGE_ERRORS = {
    "gaussian_noise": 0.886,
    "shot_noise": 0.894,
    "impulse_noise": 0.923,
    "defocus_blur": 0.820,
    "glass_blur": 0.826,
    "motion_blur": 0.786,
    "zoom_blur": 0.798,
    "snow": 0.867,
    "frost": 0.827,
    "fog": 0.819,
    "brightness": 0.565,
    "contrast": 0.853,
    "elastic_transform": 0.646,
    "pixelate": 0.718,
    "jpeg_compression": 0.607,
}
ALEXNET_AVERAGE_ERR_CLEAN = 0.4348
IMAGENET_C_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
]


@base.registry.register("imagenet_variants")
class ImagenetVariantsReport(base.Report):
  """Aggregated statistics over the ImageNet variants.

  This report contains the following ImageNet variants:
    * imagenet
    * imagenet_a,
    * imagenet_v2 (all variants)
    * imagenet_c (all variants)
  For each dataset, we compute accuracy, expected calibration
  error (ece), log-likelihood, Brier, timing, and adaptive ECE.
  """

  def __init__(self):
    super().__init__()
    self._corruption_metrics = collections.defaultdict(list)
    self._accuracy_per_corruption = collections.defaultdict(list)
    self._results = {}

  @property
  def required_measurements(self):
    def _yield_classification_specs(dataset):
      for metric in ["accuracy", "ece", "nll", "brier", "timing",
                     "adaptive_ece(datapoints_per_bin=100,threshold=0.0)"]:
        yield base.MeasurementSpec(dataset, metric)

    yield from _yield_classification_specs("imagenet")
    yield from _yield_classification_specs("imagenet_a")
    yield from _yield_classification_specs("imagenet_r")

    severities = list(range(1, 6))
    for corruption_type in IMAGENET_C_CORRUPTIONS:
      for severity in severities:
        dataset = (f"imagenet_c(corruption_type={corruption_type!r},"
                   f"severity={severity})")
        yield from _yield_classification_specs(dataset)
    yield from _yield_classification_specs(
        "imagenet_v2(variant='MATCHED_FREQUENCY')")


  def add_measurement(self, dataset_spec, metric_name, metric_results):
    dataset_name, _, dataset_kwargs = registry.parse_name_and_kwargs(
        dataset_spec)
    if dataset_name == "imagenet_c":
      if metric_name == "timing":
        value = metric_results["mean"]
      else:
        # All remaining metrics are a dictionary with a single value, with
        # the key equal to the metric name.
        value, = list(metric_results.values())
      self._corruption_metrics[f"imagenet_c/{metric_name}"].append(value)
      corruption_type = dataset_kwargs["corruption_type"]
      if metric_name == "accuracy":
        self._accuracy_per_corruption[corruption_type].append(value)
    elif dataset_name == "imagenet_v2":
      variant = dataset_kwargs["variant"]
      if variant == "MATCHED_FREQUENCY":
        if metric_name == "timing":
          value = metric_results["mean"]
        else:
          # All remaining metrics are a dictionary with a single value, with
          # the key equal to the metric name.
          value, = list(metric_results.values())
        self._results[f"imagenet_v2/{metric_name}"] = value
      else:
        logging.info("Ignoring v2 variant %r", variant)
    else:
      if metric_name == "timing":
        value = metric_results["mean"]
      else:
        # All remaining metrics produce a dictionary with a single value, with
        # the key equal to the metric name.
        value, = list(metric_results.values())
      self._results[f"{dataset_name}/{metric_name}"] = value

  def result(self) -> Dict[str, float]:
    # TODO(josipd): Check if all measurements are in.
    results = dict(self._results)
    results.update(base.compute_stats_per_bucket(self._corruption_metrics))
    ratios = []
    relative_ratios = []
    # We need errors instead of accuracies, so error = 1 - accuracy.
    try:
      clean_err = 1 - self._results["imagenet/accuracy"]
    except KeyError:
      raise base.MeasurementMissingError("Missing measurement: imagenet")
    for corruption in IMAGENET_C_CORRUPTIONS:
      if len(self._accuracy_per_corruption[corruption]) != 5:
        raise base.MeasurementMissingError(
            f"Not enough measurements for imagenet_c({corruption})")
      average_err = 1 - np.mean(
          self._accuracy_per_corruption[corruption])
      alexnet_average_err = ALEXENET_AVERAGE_ERRORS[corruption]
      ratios.append(average_err / alexnet_average_err)
      results[f"imagenet_c/{corruption}/ce"] = ratios[-1]
      relative_ratios.append((average_err - clean_err) /
                             (alexnet_average_err - ALEXNET_AVERAGE_ERR_CLEAN))
      results[f"imagenet_c/{corruption}/relative_ce"] = relative_ratios[-1]
    results["imagenet_c/mce"] = np.mean(ratios)
    results["imagenet_c/relative_mce"] = np.mean(relative_ratios)
    return results
