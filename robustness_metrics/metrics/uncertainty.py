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
"""Metrics that take into account the predicted uncertainty."""

from typing import Dict
from absl import logging
import numpy as np
from robustness_metrics.metrics import base as metrics_base
import tensorflow as tf
import tensorflow_probability as tfp
import uncertainty_metrics as um


@metrics_base.registry.register("ece")
class ExpectedCalibrationError(metrics_base.KerasMetric):
  """Expected calibration error."""

  def __init__(self, dataset_info):
    metric = um.ExpectedCalibrationError()
    super().__init__(
        dataset_info, metric, "ece", take_argmax=False, one_hot=False)


@metrics_base.registry.register("nll")
class NegativeLogLikelihood(metrics_base.KerasMetric):
  r"""Multi-class negative log likelihood.

  If the true label is k, while the predicted vector of probabilities is
  [p_1, ..., p_K], then the negative log likelihood is -log(p_k).
  """

  def __init__(self, dataset_info):
    metric = tf.keras.metrics.SparseCategoricalCrossentropy()
    super().__init__(
        dataset_info, metric, "nll", take_argmax=False, one_hot=False)


@metrics_base.registry.register("brier")
class Brier(metrics_base.KerasMetric):
  r"""Brier score.

  If the true label is k, while the predicted vector of probabilities is
  [y_1, ..., y_n], then the Brier score is equal to

    \sum_{i != k} y_i^2 + (y_k - 1)^2.
  """

  def __init__(self, dataset_info):
    metric = tf.keras.metrics.MeanSquaredError()
    super().__init__(
        dataset_info, metric, "brier", take_argmax=False, one_hot=True)

  def result(self):
    return {"brier": self._num_classes * float(self._metric.result())}


@metrics_base.registry.register("adaptive_ece")
class AdaptiveRMSECE(metrics_base.Metric):
  """Adaptive expected calibration error."""

  def __init__(
      self, dataset_info, datapoints_per_bin: int, threshold: float,
      temperature_scaling: bool = False):
    self._ids_seen = set()
    self._predictions = []
    self._labels = []
    self._datapoints_per_bin = datapoints_per_bin
    self._threshold = threshold
    self._temperature_scaling = temperature_scaling
    super().__init__(dataset_info)

  def add_predictions(self, model_predictions) -> None:
    if model_predictions.element_id is not None:
      element_id = int(model_predictions.element_id)
      if element_id in self._ids_seen:
        raise ValueError(f"You added element id {element_id!r} twice.")
      else:
        self._ids_seen.add(element_id)
    # If multiple predictions are present for a datapoint, average them:
    predictions = np.mean(np.stack(model_predictions.predictions), axis=0)
    label = model_predictions.metadata["label"]
    self._predictions.append(predictions)
    self._labels.append(label)

  def result(self) -> Dict[str, float]:
    labels = np.array(self._labels)
    predictions = np.array(self._predictions)
    if self._temperature_scaling:
      predictions = _scale_predictions(predictions, labels)
    adaptive_ece = um.numpy.gce(
        labels=labels,
        probs=predictions,
        binning_scheme="adaptive",
        datapoints_per_bin=self._datapoints_per_bin,
        class_conditional=True,
        max_prob=False,
        threshold=self._threshold,
        norm="l1",  # L1 is default in the uncertainty_metrics implementation.
    )
    return {"adaptive_ece": adaptive_ece}


def _scale_predictions(predictions, labels):
  """Performs temperature scaling on predictions to improve calibration.

  Based on Guo et al., 2017 (https://arxiv.org/abs/1706.04599).

  Args:
    predictions: Array of predictions (post-softmax) with shape
      [num_examples, num_classes].
    labels: Array of corresponding labels with shape [num_examples].

  Returns:
    Array of temperature-scaled predictions (post-softmax).
  """

  # Taking the log will recover the model output logits up to an additive
  # constant, which will not affect the results:
  logits = np.log(predictions + np.finfo(type(predictions[0][0])).eps)

  def objective(beta):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits * beta, labels=labels)
    return tf.reduce_sum(cross_entropy)

  optim_results = tfp.optimizer.lbfgs_minimize(
      tf.function(lambda beta: tfp.math.value_and_gradient(objective, beta)),
      initial_position=[1.0])

  if not optim_results.converged:
    logging.warn("LBFG-S did not converge during temperature scaling!")

  beta = optim_results.position[0]

  return tf.nn.softmax(beta * logits, axis=-1).numpy()
