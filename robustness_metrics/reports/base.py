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

"""Robustness reports.

This module provides a set of reports, which accept the values of the metrics
on the datasets and compute a robustness report.
```
report = reports.get("imagenet-c")()
report.add_metric_measurement("imagenet-c-subset-1",
                              "accuracy",
                              {"top1": .67})
results = report.results()
print(f"Results: {results!r}")
```
"""
import abc
import collections
from typing import Dict, Text, List
import dataclasses
import numpy as np
from robustness_metrics.common import registry


def compute_stats_per_bucket(
    results: Dict[str, List[float]]) -> Dict[str, float]:
  """Creates a new dictionary holding the stastistics of the value lists.

  Specifically, for each key `key`, the returned dictionary holds three keys
  `key/mean`, `key/std` and `key/count` holding the corresponding statistics.

  Args:
    results: The input dictionary.
  Returns:
    A dictionary with aggregate statistics.
  """
  results_stats = {}
  for key, values in results.items():
    results_stats[f"{key}/mean"] = np.mean(values)
    results_stats[f"{key}/std"] = np.std(values)
    results_stats[f"{key}/count"] = len(values)
  return results_stats


@dataclasses.dataclass(frozen=True)
class MeasurementSpec:
  dataset_name: Text
  metric_name: Text


class MeasurementMissingError(Exception):
  """Raised when the result cannot be computed as measurements are missing."""


class Report(metaclass=abc.ABCMeta):
  """A report computes scores from multiple metric evaluations.

  Each report specifies in `required_measurements` which metrics on which
  datasets have to be computed to be able to produce its scores. The
  measurements are fed using `add_measurement`. Finally, the method `result`
  computes the final scores.
  """

  @property
  def required_measurements(self) -> List[MeasurementSpec]:
    """The collection of measurements that have to be computed."""
    return []

  @abc.abstractmethod
  def add_measurement(self, dataset_name: Text, metric_name: Text,
                      metric_results: Dict[Text, float]):
    """Adds the results from computing the metric on the dataset.

    Args:
      dataset_name: The identifier of the dataset on which the metric was
        computed.
      metric_name: The name of the metric.
      metric_results: The result from the metric computation.
    """

  @abc.abstractmethod
  def result(self) -> Dict[Text, float]:
    """Computes the results from the given measurements.

    Returns:
      A dictionary mapping the name of each score to its value.
    """


registry = registry.Registry(Report)
get = registry.get


class UnionReport(Report):
  """Concatenates the required measurements in a single dictionary.

  Specifically, if the metric `metric_name` computed a value of `value` with
  key `key` on dataset `dataset_name`, the report will report a value of
  `value` under the name `dataset_name/metric_name/key`.
  """

  def __init__(self):
    self._metric_measurements = collections.defaultdict()
    self._metrics_seen = set()

  def add_measurement(self, dataset_name: Text, metric_name: Text,
                      metric_results: Dict[Text, float]):
    spec = MeasurementSpec(dataset_name=dataset_name, metric_name=metric_name)
    if spec not in self.required_measurements:
      raise ValueError(f"Adding an unknown observation: {spec!r}")
    self._metrics_seen.add(spec)

    for key, value in metric_results.items():
      self._metric_measurements[f"{dataset_name}/{metric_name}/{key}"] = value

  def result(self) -> Dict[Text, float]:
    if set(self.required_measurements) != self._metrics_seen:
      raise ValueError("Observed: {!r}, Required: {!r}".format(
          self._metrics_seen, self.required_measurements))
    return self._metric_measurements


@registry.register("classification_report")
class ClassficationReport(UnionReport):
  """Computes commonly used classification metrics.

  The model should compute the class probabilities (not logits). This report
  will compute accuracy, expected calibration error, negative log-likelihood,
  and Brier scores of the predictions.
  """

  def __init__(self, datasets):
    self._datasets = datasets
    super().__init__()

  @property
  def required_measurements(self):
    metrics = ["accuracy", "ece", "nll", "brier"]
    for dataset in self._datasets:
      for metric in metrics:
        yield MeasurementSpec(dataset, metric)


@registry.register("ensemble_classification_report")
class EnsembleClassficationReport(ClassficationReport):
  """Complete ClassficationReport with ensemble-specific metrics.

  In particular, we add diversity metrics useful to compare ensemble members.
  """

  @property
  def required_measurements(self):
    yield from super().required_measurements

    for dataset in self._datasets:
      yield MeasurementSpec(
          dataset, "average_pairwise_diversity(normalize_disagreement=True)")
      yield MeasurementSpec(
          dataset, "average_pairwise_diversity(normalize_disagreement=False)")
