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
"""Robustness metrics.

This module provides a set of metrics, which accept `ModelPrediction`s and
compute a dictionary of floating point numbers. Each metric is registered under
a unique identifier, and the method `get` can be used to obtain the Metric
subclass given the identifier. Example usage:

```
import robustness_metrics as rm

metric = rm.metrics.get("accuracy")  # or rm.metrics.Accuracy()
metric.add_predictions(
  ModelPredictions(metadata={"label": 1, "element_id": 2},
                   predictions=[[.2, .8], [.7, .3]]))
results = model.results()
print(f"Results: {results!r}")
```
"""
import operator
from typing import Text, Type

from robustness_metrics.common import types
from robustness_metrics.metrics import base
from robustness_metrics.metrics import retrieval
from robustness_metrics.metrics import serialization
from robustness_metrics.metrics import timing
from robustness_metrics.metrics import uncertainty
from robustness_metrics.metrics.base import Accuracy
from robustness_metrics.metrics.base import AggregatedAccuracy
from robustness_metrics.metrics.base import FullBatchMetric
from robustness_metrics.metrics.base import KerasMetric
from robustness_metrics.metrics.base import Metric
from robustness_metrics.metrics.base import Precision
from robustness_metrics.metrics.base import registry
from robustness_metrics.metrics.base import TopKAccuracy
from robustness_metrics.metrics.diversity import AveragePairwiseDiversity
from robustness_metrics.metrics.information_criteria import EnsembleCrossEntropy
from robustness_metrics.metrics.information_criteria import GibbsCrossEntropy
from robustness_metrics.metrics.retrieval import AucPr
from robustness_metrics.metrics.retrieval import AucRoc
from robustness_metrics.metrics.retrieval import FalsePositiveRate95
from robustness_metrics.metrics.serialization import Serializer
from robustness_metrics.metrics.synthetic import Synthetic
from robustness_metrics.metrics.timing import TimingStatsMetric
from robustness_metrics.metrics.uncertainty import AdaptiveCalibrationError
from robustness_metrics.metrics.uncertainty import Brier
from robustness_metrics.metrics.uncertainty import BrierDecomposition
from robustness_metrics.metrics.uncertainty import CalibrationAUC
from robustness_metrics.metrics.uncertainty import CRPSSCore
from robustness_metrics.metrics.uncertainty import ExpectedCalibrationError
from robustness_metrics.metrics.uncertainty import GeneralCalibrationError
from robustness_metrics.metrics.uncertainty import IsotonicRegression
from robustness_metrics.metrics.uncertainty import MonotonicSweepCalibrationError
from robustness_metrics.metrics.uncertainty import NegativeLogLikelihood
from robustness_metrics.metrics.uncertainty import OracleCollaborativeAccuracy
from robustness_metrics.metrics.uncertainty import OracleCollaborativeAUC
from robustness_metrics.metrics.uncertainty import RootMeanSquaredCalibrationError
from robustness_metrics.metrics.uncertainty import SemiParametricCalibrationError
from robustness_metrics.metrics.uncertainty import SemiParametricCalibrationErrorConfidenceInterval
from robustness_metrics.metrics.uncertainty import StaticCalibrationError
from robustness_metrics.metrics.uncertainty import TemperatureScaling
from robustness_metrics.metrics.uncertainty import ThresholdedAdaptiveCalibrationError


def get(metric_name: Text, dataset_info=None):
  """Returns the subclass of `Metric` that has the given name.

  Args:
    metric_name: The name of the metric.
    dataset_info: The dataset info to be passed to the metric initializer.
  Returns:
    The subclass if `Metric` that has the given name.
  Raises:
    KeyError: If there is no metric registered under the given name.
  """
  # Register your metric by adding an entry in the dictionary below.
  return base.registry.get_instance(metric_name, dataset_info=dataset_info)


def _recursive_map(fn, dict_or_val):
  if isinstance(dict_or_val, dict):
    return {k: _recursive_map(fn, v) for k, v in dict_or_val.items()}
  else:
    return fn(dict_or_val)


def add_batch(metric: base.Metric, predictions, **metadata):
  """Add a batch of predictions.

  Example usage:
  ```
  metric = rm.metrics.get("accuracy")()
  rm.metrics.add_batch(metric, [[.6, .4], [.9, .1]], label=[1, 0])
  metric.result()  # Returns {"accuracy": 0.5}.
  ```

  Args:
    metric: The metric where the predictions will be added.
    predictions: A 2d array (list or numpy array), containing one prediction per
      row.
    **metadata:
      The keys and values that will be used to construct the metadata. It can
      be any (arbitrarily) nested dictionary, with 2d arrays (list or numpy
      arrays) leaves, each holding one example per row.
  """
  for i, predictions_i in enumerate(predictions):
    metadata_i = _recursive_map(operator.itemgetter(i), metadata)
    metric.add_predictions(types.ModelPredictions(predictions=[predictions_i]),
                           metadata_i)


__all__ = [
    "Accuracy",
    "AdaptiveCalibrationError",
    "AggregatedAccuracy",
    "AucPr",
    "AucRoc",
    "AveragePairwiseDiversity",
    "Brier",
    "BrierDecomposition",
    "CRPSSCore",
    "EnsembleCrossEntropy",
    "ExpectedCalibrationError",
    "FalsePositiveRate95",
    "FullBatchMetric",
    "GeneralCalibrationError",
    "GibbsCrossEntropy",
    "ImageNetVidRobust",
    "IsotonicRegression",
    "KerasMetric",
    "Metric",
    "MonotonicSweepCalibrationError",
    "NegativeLogLikelihood",
    "ObjectNetAccuracy",
    "ObjectNetGCE",
    "ObjectNetMetric",
    "Precision",
    "RootMeanSquaredCalibrationError",
    "SemiParametricCalibrationError",
    "SemiParametricCalibrationErrorConfidenceInterval",
    "Serializer",
    "StaticCalibrationError",
    "Synthetic",
    "TemperatureScaling",
    "ThresholdedAdaptiveCalibrationError",
    "TimingStatsMetric",
    "TopKAccuracy",
    "add_batch",
    "base",
    "get",
    "registry",
    "retrieval",
    "serialization",
    "timing",
    "uncertainty",
]
