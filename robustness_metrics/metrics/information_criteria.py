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

"""Information criteria."""

from typing import Dict, Text

from robustness_metrics.common import types
from robustness_metrics.metrics import base as metrics_base
import tensorflow as tf


class EnsembleCrossEntropy(metrics_base.Metric):
  """Cross-entropy of an ensemble distribution.

  For each datapoint (x,y), the ensemble's negative log-probability is:

  ```
  -log p(y|x) = -log sum_{m=1}^{ensemble_size} exp(log p(y|x,theta_m)) +
                log ensemble_size.
  ```

  The cross entropy is the expected negative log-probability with respect to
  the true data distribution.
  """

  def __init__(self, dataset_info=None, binary=False, aggregate=True):
    """Init.

    Args:
      dataset_info: Unused.
      binary: bool, whether it is a binary classification (sigmoid as
        activation).
      aggregate: bool, whether or not to average over the batch.
    """
    del dataset_info
    self._labels = []
    self._logits = []
    self._binary = binary
    self._aggregate = aggregate

  def add_predictions(self,
                      model_predictions: types.ModelPredictions,
                      metadata: types.Features) -> None:
    self.add_batch(
        model_predictions.predictions,
        labels=metadata['labels'])

  def add_batch(self, model_predictions, *, labels=None) -> None:
    """Adds a batch of predictions for a batch of examples.

    Args:
      model_predictions: logits of shape [ensemble_size, ..., num_classes]. Note
        that unlike some other metrics, this metric takes unnormalized logits
        instead of probabilities.
      labels: tf.Tensor of shape [...].
    """
    self._logits.append(model_predictions)
    self._labels.append(labels)

  def result(self) -> Dict[Text, float]:
    logits = tf.concat(self._logits, axis=0)
    labels = tf.concat(self._labels, axis=0)
    ensemble_size = float(logits.shape[0])
    if self._binary:
      ce = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.broadcast_to(labels[tf.newaxis, ...], tf.shape(logits)),
          logits=logits)
    else:
      labels = tf.cast(labels, tf.int32)
      labels = tf.broadcast_to(labels[tf.newaxis, ...], tf.shape(logits)[:-1])
      ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)
    nll = -tf.reduce_logsumexp(-ce, axis=0) + tf.math.log(ensemble_size)
    if self._aggregate:
      nll = tf.reduce_mean(nll)
    return {'negative_log_likelihood': nll}


class GibbsCrossEntropy(metrics_base.Metric):
  """Average cross entropy for ensemble members (Gibbs cross entropy).

  For each datapoint (x,y), the ensemble's Gibbs cross entropy is:

  ```
  - (1/ensemble_size) sum_{m=1}^ensemble_size log p(y|x,theta_m).
  ```

  The Gibbs cross entropy approximates the average cross entropy of a single
  model drawn from the (Gibbs) ensemble.
  """

  def __init__(self, dataset_info=None, binary=False, aggregate=True):
    """Init.

    Args:
      dataset_info: Unused.
      binary: bool, whether it is a binary classification (sigmoid as
        activation).
      aggregate: bool, whether or not to average over the batch.
    """
    del dataset_info
    self._labels = []
    self._logits = []
    self._binary = binary
    self._aggregate = aggregate

  def add_predictions(self,
                      model_predictions: types.ModelPredictions,
                      metadata: types.Features) -> None:
    self.add_batch(
        model_predictions.predictions,
        labels=metadata['labels'])

  def add_batch(self, model_predictions, *, labels=None) -> None:
    """Adds a batch of predictions for a batch of examples.

    Args:
      model_predictions: logits of shape [ensemble_size, ..., num_classes]. Note
        that unlike some other metrics, this metric takes unnormalized logits
        instead of probabilities.
      labels: tf.Tensor of shape [...].
    """
    self._logits.append(model_predictions)
    self._labels.append(labels)

  def result(self) -> Dict[Text, float]:
    logits = tf.concat(self._logits, axis=0)
    labels = tf.concat(self._labels, axis=0)
    if self._binary:
      nll = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.broadcast_to(labels[tf.newaxis, ...], tf.shape(logits)),
          logits=logits)
    else:
      labels = tf.cast(labels, tf.int32)
      labels = tf.broadcast_to(labels[tf.newaxis, ...], tf.shape(logits)[:-1])
      nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels,
          logits=logits)
    nll = tf.reduce_mean(nll, axis=0)
    if self._aggregate:
      nll = tf.reduce_mean(nll)
    return {'gibbs_cross_entropy': nll}

