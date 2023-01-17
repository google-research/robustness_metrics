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

"""Report of the metrics of the Plex paper https://arxiv.org/abs/2207.07411."""
# TODO(rjenatton): Add the CifarPlexReport.

from robustness_metrics.common import registry
from robustness_metrics.reports import base
from robustness_metrics.reports import imagenet_variants
from robustness_metrics.reports import ood_detection


@base.registry.register("imagenet_plex")
class ImagenetPlexReport(imagenet_variants.ImagenetVariantsReport):
  """ImageNet report of the Plex paper https://arxiv.org/abs/2207.07411."""

  def __init__(self, convert_imagenet_real_labels: bool = True):
    super().__init__()
    self._convert_imagenet_real_labels = convert_imagenet_real_labels
    self._ood_report = ood_detection.ImagenetOodDetectionReport()

  def _yield_selective_prediction_metrics(self, use_dataset_labelset=None):
    if use_dataset_labelset is None:
      labelset_str = ""
    else:
      labelset_str = f"use_dataset_labelset={use_dataset_labelset},"
    for t in [0.005, 0.01, 0.02, 0.05]:
      yield (f"collaborative_auc("
             f"{labelset_str}"
             f"oracle_fraction={t},"
             f"num_bins=1000,"
             f"default_binning_score_to_confidence=True,"
             f"take_argmax=True,"
             f"key_name='collaborative_auc@{t}'"
             f")")

  def _yield_metrics_to_evaluate(self, use_dataset_labelset=None):
    """Yields metrics to be evaluated."""
    yield from super()._yield_metrics_to_evaluate(
        use_dataset_labelset=use_dataset_labelset)
    yield from self._yield_selective_prediction_metrics(
        use_dataset_labelset=use_dataset_labelset)

  @property
  def required_measurements(self):
    yield from super().required_measurements
    for m in self._ood_report.required_measurements:
      # This is the OOD convention used in Plex.
      if "ood_with_positive_labels=True" in m.dataset_name:
        yield m

  def add_measurement(self, dataset_spec, metric_name, metric_results):
    metrics_to_filter = [
        "auc_pr", "auc_roc", "fpr95", "nll(soft_labels=True)",
        "collaborative_auc"
    ]
    if any(m in metric_name for m in metrics_to_filter):
      dataset_name, _, dataset_kwargs = registry.parse_name_and_kwargs(
          dataset_spec)

      if dataset_name == "imagenet_v2":
        if dataset_kwargs["variant"] != "MATCHED_FREQUENCY":
          return

      for metric_key, metric_value in metric_results.items():
        key = f"{dataset_name}/{metric_key}"
        if dataset_name == "imagenet_c":
          self._corruption_metrics[key].append(metric_value)
        else:
          self._results[key] = metric_value
    else:
      super().add_measurement(dataset_spec, metric_name, metric_results)
