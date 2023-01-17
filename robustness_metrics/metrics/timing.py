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

"""Metrics that report the inference speed of the model."""

from typing import Dict
import numpy as np
from robustness_metrics.metrics import base as metrics_base


@metrics_base.registry.register("timing")
class TimingStatsMetric(metrics_base.Metric):
  """Computes the mean inference speed and several quantiles.

  The result has the following keys:
    "mean": Holding the mean inference time.
    "quantile/{x}": The x-th quantile of the inference times.
  """

  def __init__(self, dataset_info):
    self._time_deltas_in_s = []
    super().__init__(dataset_info)

  def add_predictions(self, model_predictions, metadata) -> None:
    self._time_deltas_in_s.append(model_predictions.time_in_s)

  def result(self) -> Dict[str, float]:
    result = {"mean": np.mean(self._time_deltas_in_s)}
    quantiles = [0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95]
    quantiles_values = np.quantile(self._time_deltas_in_s, quantiles)
    for quantile, value in zip(quantiles, quantiles_values):
      result[f"quantile/{quantile}"] = value
    return result
