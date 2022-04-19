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

"""Retrieval metrics such as AUC-ROC and AUC-PR."""

import abc
from typing import Dict

import numpy as np
from robustness_metrics.metrics import base as metrics_base
import sklearn.metrics


def _format_predictions(predictions: np.ndarray,
                        is_binary_classification: bool):
  """Check and format the tensor of predictions."""
  if is_binary_classification:
    assert_msg = ("Expect binary classification: predictions must have shape "
                  f" (dataset size, 2); received {predictions.shape}.")
    assert predictions.ndim == 2 == predictions.shape[1], assert_msg
    # In the binary classification case, the retrieval metrics expect the
    # predictions to be that of the positive class (i.e., with index 1).
    predictions = predictions[:, 1]
  else:
    # In the multiclass case, we take the maximum predictions across classes.
    # This is motivated by the OOD detection setting with the standard approach:
    #   "Maximum over softmax probabilities" (MSP),
    #   See https://arxiv.org/pdf/2106.03004.pdf.
    # TODO(rjenatton): Generalize this logic to other known schemes, e.g.,
    #                  entropy(predictions, axis=-1) or Mahalanobis distance.
    predictions = np.max(predictions, axis=-1)
  return predictions


class BinaryRetrievalMetric(
    metrics_base.FullBatchMetric, metaclass=abc.ABCMeta):
  """Abstract class for binary retrieval metrics such as AUC-PR and AUC-ROC."""

  def __init__(self, dataset_info=None, is_binary_classification=False):
    super().__init__(dataset_info)
    self.is_binary_classification = is_binary_classification

  @abc.abstractmethod
  def _get_metric(self, labels: np.ndarray,
                  predictions: np.ndarray) -> Dict[str, float]:
    """Compute the metric.

    Args:
      labels: array of labels.
      predictions: array of predictions.
    Returns
      A dictionary of the form {metric name: metric value, ...}.
    """

  def result(self) -> Dict[str, float]:
    labels = np.asarray(self._labels)
    predictions = np.asarray(self._predictions)
    predictions = _format_predictions(predictions,
                                      self.is_binary_classification)
    assert_msg = ("Labels/predictions do not match; respective shapes"
                  f" {labels.shape} and {predictions.shape}.")
    assert len(labels) == len(predictions), assert_msg
    return self._get_metric(labels, predictions)


@metrics_base.registry.register("auc_pr")
class AucPr(BinaryRetrievalMetric):
  """Area under the precision and recall curve.

  Based on the implementation of:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
  """

  def _get_metric(self, labels: np.ndarray,
                  predictions: np.ndarray) -> Dict[str, float]:
    return {
        "auc_pr": sklearn.metrics.average_precision_score(labels, predictions)
    }


@metrics_base.registry.register("auc_roc")
class AucRoc(BinaryRetrievalMetric):
  """Area under the receiver operating characteristic curve.

  Based on the implementation of:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
  """

  def _get_metric(self, labels: np.ndarray,
                  predictions: np.ndarray) -> Dict[str, float]:
    return {
        "auc_roc": sklearn.metrics.roc_auc_score(labels, predictions)
    }


@metrics_base.registry.register("fpr95")
class FalsePositiveRate95(BinaryRetrievalMetric):
  """False positive rate at 95% true positive rate.

  Based on the implementation of:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
  """

  def _get_metric(self, labels: np.ndarray,
                  predictions: np.ndarray) -> Dict[str, float]:
    threshold = 0.95
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, predictions)
    return {
        "fpr95": fpr[np.argmax(tpr >= threshold)]
    }
