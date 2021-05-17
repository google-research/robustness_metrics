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

r"""Reports that evaluate models over the ImageNet label space.

"""

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

  def _yield_metrics_to_evaluate(self, use_dataset_labelset=None):
    """Yields metrics to be evaluated."""
    metrics = ["accuracy", "nll", "brier"]
    if use_dataset_labelset is not None:
      metrics = [f"{metric}(use_dataset_labelset={use_dataset_labelset})"
                 for metric in metrics]
    yield from metrics

  def _yield_classification_specs(self, dataset, use_dataset_labelset=None):
    """Yields a MeasurementSpec for each metric and a given dataset."""
    for metric in self._yield_metrics_to_evaluate(
        use_dataset_labelset=use_dataset_labelset):
      yield base.MeasurementSpec(dataset, metric)

  @property
  def required_measurements(self):

    yield from self._yield_classification_specs("imagenet")
    yield from self._yield_classification_specs("imagenet_a")
    yield from self._yield_classification_specs("imagenet_r")

    severities = list(range(1, 6))
    for corruption_type in IMAGENET_C_CORRUPTIONS:
      for severity in severities:
        dataset = (f"imagenet_c(corruption_type={corruption_type!r},"
                   f"severity={severity})")
        yield from self._yield_classification_specs(dataset)
    yield from self._yield_classification_specs(
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
      if len(metric_results) == 1:
        # Metrics produce a dictionary with a single value, with the key equal
        # to the metric name.
        value, = list(metric_results.values())
      elif metric_name == "timing":
        value = metric_results["mean"]
      else:
        raise ValueError(
            f"Must specify which key to use: {list(metric_results.keys())}")
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


@base.registry.register("imagenet_variants_gce_sweep")
class ImagenetVariantsGceSweepReport(ImagenetVariantsReport):
  """Sweeps over additional GCE variants."""

  def _yield_gce_metrics(self, use_dataset_labelset=None):
    # Add a sweep over General Calibration Errors:
    max_prob = True  # Whether to consider the max prediction or all.
    class_conditional = False  # Wheter to consider classes individually.
    threshold = 0.0  # Predicted probablities less than `threshold` are removed.
    recalibration_method = None  # Post-hoc recalibration on full dataset.
    if use_dataset_labelset is None:
      labelset_str = ""
    else:
      labelset_str = f"use_dataset_labelset={use_dataset_labelset},"
    for norm in ["l1", "l2"]:  # Which distance to use for the CE computation.
      for binning_scheme in ["adaptive", "even"]:
        for num_bins in [15, 100]:
          yield (f"gce("
                 f"binning_scheme={binning_scheme!r},"
                 f"max_prob={max_prob},"
                 f"class_conditional={class_conditional},"
                 f"norm='{norm}',"
                 f"num_bins={num_bins},"
                 f"threshold={threshold},"
                 f"recalibration_method={recalibration_method},"
                 f"{labelset_str}"
                 f")")

  def _yield_metrics_to_evaluate(self, use_dataset_labelset=None):
    """Yields metrics to be evaluated."""

    # Include standard metrics:
    yield from super()._yield_metrics_to_evaluate(
        use_dataset_labelset=use_dataset_labelset)
    yield from self._yield_gce_metrics(
        use_dataset_labelset=use_dataset_labelset)

  @property
  def required_measurements(self):
    yield from super().required_measurements
    yield base.MeasurementSpec("objectnet", "objectnet_accuracy")
    for gce_spec in self._yield_gce_metrics():
      yield base.MeasurementSpec("objectnet", f"objectnet_{gce_spec}")


@base.registry.register("imagenet_rescaling")
class ImagenetRescalingReport(ImagenetVariantsReport):
  """Report that computes rescaling parameters for ImageNet."""

  @property
  def required_measurements(self):
    rescaling_methods = ["temperature_scaling"]
    for rescaling_method in rescaling_methods:
      metric = f"{rescaling_method}"
      yield base.MeasurementSpec("imagenet(split='validation[:20%]')", metric)
      yield base.MeasurementSpec("imagenet(split='validation[20%:]')", metric)


@base.registry.register("imagenet_c_rescaling")
class ImagenetCRescalingReport(ImagenetVariantsReport):
  """Report that computes rescaling parameters for ImageNet-C.

  This is not used for actual rescaling, but for over/under-confidence analysis.
  """

  @property
  def required_measurements(self):
    rescaling_methods = ["temperature_scaling"]
    severities = list(range(1, 6))
    for rescaling_method in rescaling_methods:
      metric = f"{rescaling_method}"
      for corruption_type in IMAGENET_C_CORRUPTIONS:
        for severity in severities:
          dataset = (f"imagenet_c(corruption_type={corruption_type!r},"
                     f"severity={severity},"
                     "split='validation[:20%]')")
          dataset = (f"imagenet_c(corruption_type={corruption_type!r},"
                     f"severity={severity},"
                     "split='validation[20%:]')")
          yield base.MeasurementSpec(dataset, metric)


@base.registry.register("ood_rescaling")
class OodRescalingReport(ImagenetVariantsReport):
  """Report that computes rescaling parameters for OOD datasets.

  This is not used for actual rescaling, but for over/under-confidence analysis.
  """

  @property
  def required_measurements(self):
    rescaling_methods = ["temperature_scaling"]
    for rescaling_method in rescaling_methods:
      yield base.MeasurementSpec("imagenet_a", rescaling_method)
      yield base.MeasurementSpec("imagenet_r", rescaling_method)
      yield base.MeasurementSpec(
          "imagenet_v2(variant='MATCHED_FREQUENCY')", rescaling_method)


@base.registry.register("imagenet_rescaled_variants_gce_sweep")
class ImagenetVariantsRescaledGceSweepReport(ImagenetVariantsGceSweepReport):
  """Similar to ImagenetVariantsGceSweepReport, but with [20%:] data split."""

  @property
  def required_measurements(self):
    split = "validation[20%:]"

    yield from self._yield_classification_specs(f"imagenet(split={split!r})")
    yield from self._yield_classification_specs(
        "imagenet_a", use_dataset_labelset=True)
    yield from self._yield_classification_specs(
        "imagenet_r", use_dataset_labelset=True)

    severities = list(range(1, 6))
    for corruption_type in IMAGENET_C_CORRUPTIONS:
      for severity in severities:
        dataset = (f"imagenet_c(corruption_type={corruption_type!r},"
                   f"severity={severity},"
                   f"split={split!r})")
        yield from self._yield_classification_specs(dataset)
    yield from self._yield_classification_specs(
        "imagenet_v2(variant='MATCHED_FREQUENCY')")
