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

"""Metrics for model diversity."""

import itertools
from typing import Any, Dict, Optional, Text

from robustness_metrics.common import types
from robustness_metrics.metrics import base as metrics_base
import tensorflow as tf


def disagreement(logits_1, logits_2):
  """Disagreement between the predictions of two classifiers."""
  preds_1 = tf.argmax(logits_1, axis=-1, output_type=tf.int32)
  preds_2 = tf.argmax(logits_2, axis=-1, output_type=tf.int32)
  return tf.cast(preds_1 != preds_2, tf.float32)


def kl_divergence(p, q, clip=False):
  """Generalized KL divergence [1] for unnormalized distributions.

  Args:
    p: tf.Tensor of shape [batch_size, num_classes].
    q: tf.Tensor of shape [batch_size, num_classes].
    clip: bool.

  Returns:
    tf.Tensor of shape [batch_size].

  ## References

  [1] Lee, Daniel D., and H. Sebastian Seung. "Algorithms for non-negative
  matrix factorization." Advances in neural information processing systems.
  2001.
  """
  if clip:
    p = tf.clip_by_value(p, tf.keras.backend.epsilon(), 1)
    q = tf.clip_by_value(q, tf.keras.backend.epsilon(), 1)
  return tf.reduce_sum(p * tf.math.log(p / q), axis=-1)


def cosine_distance(x, y):
  """Cosine distance between vectors x and y."""
  x_norm = tf.math.sqrt(tf.reduce_sum(tf.pow(x, 2), axis=-1))
  x_norm = tf.reshape(x_norm, (-1, 1))
  y_norm = tf.math.sqrt(tf.reduce_sum(tf.pow(y, 2), axis=-1))
  y_norm = tf.reshape(y_norm, (-1, 1))
  normalized_x = x / x_norm
  normalized_y = y / y_norm
  return tf.reduce_sum(normalized_x * normalized_y, axis=-1)


def bregman_kl_variance(x):
  """Bregman variance under a KL loss as defined in arxiv.org/abs/2202.04167.

  This function returns a value that can be either interpreted as
  1. The empirical (biased) estimate of the variance of a *single* model, where
     the variance is obtained by the generalization of the bias-variance
     decomposition to the KL divergence loss. See arxiv.org/pdf/2202.04167,
     Eq. (5) for the definion of variance and Prop. 4.1 for the bias of this
     estimator.
  2. An estimate of how diverse an ensemble is: when the ensemble output is
     obtained by averaging individual predictions in probability space, the
     expected loss of the ensemble is the average loss of each individual model
     minus this diversity term. As such, increasing the diversity (while keeping
     the mean individual loss fixed) reduces ensemble loss. See, e.g.,
     arxiv.org/pdf/1902.04422 for this term as a diversity regularizer.

  Args:
    x: tf.Tensor of probabilities, of shape [ensemble_size, batch_size=1, 1,
      num_classes].

  Returns:
    tf.Tensor of shape [batch_size].
  """
  num_models = tf.shape(x)[0]
  batch_size = tf.shape(x)[1]

  variance = tf.zeros(batch_size)
  central_prediction = tf.nn.softmax(tf.reduce_mean(tf.math.log(x), axis=0))
  for i in range(num_models):
    variance += kl_divergence(central_prediction, x[i])
  return variance / tf.cast(num_models, dtype=tf.float32)


@metrics_base.registry.register('average_pairwise_diversity')
class AveragePairwiseDiversity(metrics_base.Metric):
  """Average pairwise diversity of models.

  The metric computes

  ```
  1/N sum_{n=1}^dataset_size
      1/num_models**2 sum_{m,m'=1}^num_models div(p_{n, m}, p_{n, m'})
  ```

  for each diversity measure `div`. Given a batch, the metric computes

  ```
  sum_{b=1}^batch_size
      1/num_models**2 sum_{m,m'=1}^num_models div(p_{b, m}, p_{b, m'}).
  ```

  This quantity is accumulated across each call to `add_batch()`. `result()`
  then averages the quantity by the total number of data points.
  """

  def __init__(self, dataset_info=None, normalize_disagreement=False):
    """Initializes the metric.

    Args:
      dataset_info: The DatasetInfo object associated with the dataset. Unused.
      normalize_disagreement: Whether to normalize the average disagreement by
        classification error. Normalizing disagreement adjusts for the fact that
        inaccurate models have a higher potential of disagreeing because wrong
        predictions may be more random.
    """
    del dataset_info
    self._normalize_disagreement = normalize_disagreement
    self._accuracy = None
    if self._normalize_disagreement:
      self._accuracy = metrics_base.Accuracy()
    self._dataset_size = tf.Variable(0,
                                     trainable=False,
                                     aggregation=tf.VariableAggregation.SUM)
    self._disagreement = tf.Variable(0.,
                                     trainable=False,
                                     aggregation=tf.VariableAggregation.SUM)
    self._kl_divergence = tf.Variable(0.,
                                      trainable=False,
                                      aggregation=tf.VariableAggregation.SUM)
    self._cosine_distance = tf.Variable(0.,
                                        trainable=False,
                                        aggregation=tf.VariableAggregation.SUM)

    self._bregman_kl_variance = tf.Variable(
        0., trainable=False, aggregation=tf.VariableAggregation.SUM)

  def add_predictions(self,
                      model_predictions: types.ModelPredictions,
                      metadata: types.Features) -> None:
    # To update the metric on individual data points, expand the inputs to have
    # batch size 1 and call `add_batch`.
    batch_predictions = tf.expand_dims(model_predictions.predictions, axis=1)
    if 'label' in metadata:
      metadata['label'] = tf.expand_dims(metadata['label'], axis=1)
    self.add_batch(batch_predictions, **metadata)

  def add_batch(self,
                model_predictions,
                **metadata: Optional[Dict[Text, Any]]) -> None:
    """Adds a batch of predictions for a batch of examples.

    Args:
      model_predictions: Tensor-like of shape [num_models, batch_size,
        num_classes] representing the multiple predictions, one for each example
        in the batch.
      **metadata: Metadata of model predictions. Only necessary if
        normalize_disagreement is True.
    """
    model_predictions = tf.convert_to_tensor(model_predictions)
    num_models = model_predictions.shape[0]
    batch_size = tf.shape(model_predictions)[1]
    batch_disagreement = []
    batch_kl_divergence = []
    batch_cosine_distance = []
    for pair in list(itertools.combinations(range(num_models), 2)):
      probs_1 = model_predictions[pair[0]]
      probs_2 = model_predictions[pair[1]]
      batch_disagreement.append(
          tf.reduce_sum(disagreement(probs_1, probs_2)))
      batch_kl_divergence.append(
          tf.reduce_sum(kl_divergence(probs_1, probs_2)))
      batch_cosine_distance.append(
          tf.reduce_sum(cosine_distance(probs_1, probs_2)))

    # TODO(ghassen): we could also return max and min pairwise metrics.
    batch_disagreement = tf.reduce_mean(batch_disagreement)
    batch_kl_divergence = tf.reduce_mean(batch_kl_divergence)
    batch_cosine_distance = tf.reduce_mean(batch_cosine_distance)
    batch_bregman_kl_variance = tf.reduce_sum(
        bregman_kl_variance(model_predictions))

    self._dataset_size.assign_add(batch_size)
    self._disagreement.assign_add(batch_disagreement)
    self._kl_divergence.assign_add(batch_kl_divergence)
    self._cosine_distance.assign_add(batch_cosine_distance)
    self._bregman_kl_variance.assign_add(batch_bregman_kl_variance)

    if self._normalize_disagreement:
      if self._accuracy is None:
        raise ValueError('Accuracy not initialized.')
      ensemble_predictions = tf.reduce_mean(model_predictions, axis=0)
      self._accuracy.add_batch(ensemble_predictions, **metadata)

  def reset_states(self):
    if self._normalize_disagreement and self._accuracy is not None:
      self._accuracy.reset_states()
    self._dataset_size.assign(0)
    self._disagreement.assign(0.)
    self._kl_divergence.assign(0.)
    self._cosine_distance.assign(0.)
    self._bregman_kl_variance.assign(0.)

  def result(self) -> Dict[Text, float]:
    """Computes the results from all the predictions it has seen so far.

    Returns:
      A dictionary mapping the name of each computed metric to its value.
    """
    if self._dataset_size == 0:
      return {
          'disagreement': 0.0,
          'average_kl': 0.0,
          'cosine_similarity': 0.0,
          'bregman_kl_variance': 0.0,
      }

    dataset_size = tf.cast(self._dataset_size, self._disagreement.dtype)
    avg_disagreement = self._disagreement / dataset_size
    if self._normalize_disagreement and self._accuracy is not None:
      classification_error = 1. - self._accuracy.result()['accuracy']
      avg_disagreement /= classification_error + tf.keras.backend.epsilon()

    avg_kl = self._kl_divergence / dataset_size
    avg_cosine_distance = self._cosine_distance / dataset_size
    avg_bregman_kl_variance = self._bregman_kl_variance / dataset_size
    return {
        'disagreement': float(avg_disagreement),
        'average_kl': float(avg_kl),
        'cosine_similarity': float(avg_cosine_distance),
        'bregman_kl_variance': float(avg_bregman_kl_variance),
    }
