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
"""Metrics that take into account the predicted uncertainty."""

import os
import pickle
from typing import Any, Dict, Optional, List, Union
import warnings

from absl import logging
import numpy as np
from robustness_metrics.common import types
from robustness_metrics.datasets import base as datasets_base
from robustness_metrics.metrics import base as metrics_base
import scipy.interpolate
import scipy.stats
from sklearn import isotonic as sklearn_ir
import sklearn.model_selection
import tensorflow as tf
import tensorflow_probability as tfp

DEFAULT_PICKLE_PATH = "/tmp"


class _KerasECEMetric(tf.keras.metrics.Metric):
  """Expected Calibration Error.

  Expected calibration error (Guo et al., 2017, Naeini et al., 2015) is a scalar
  measure of calibration for probabilistic models. Calibration is defined as the
  level to which the accuracy over a set of predicted decisions and true
  outcomes associated with a given predicted probability level matches the
  predicted probability. A perfectly calibrated model would be correct `p`% of
  the time for all examples for which the predicted probability was `p`%, over
  all values of `p`.

  This metric can be computed as follows. First, cut up the probability space
  interval [0, 1] into some number of bins. Then, for each example, store the
  predicted class (based on a threshold of 0.5 in the binary case and the max
  probability in the multiclass case), the predicted probability corresponding
  to the predicted class, and the true label into the corresponding bin based on
  the predicted probability. Then, for each bin, compute the average predicted
  probability ("confidence"), the accuracy of the predicted classes, and the
  absolute difference between the confidence and the accuracy ("calibration
  error"). Expected calibration error can then be computed as a weighted average
  calibration error over all bins, weighted based on the number of examples per
  bin.

  Perfect calibration under this setup is when, for all bins, the average
  predicted probability matches the accuracy, and thus the expected calibration
  error equals zero. In the limit as the number of bins goes to infinity, the
  predicted probability would be equal to the accuracy for all possible
  probabilities.

  References:
    1. Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. On Calibration of Modern
       Neural Networks. in International Conference on Machine Learning (ICML)
       cs.LG, (Cornell University Library, 2017).
    2. Naeini, M. P., Cooper, G. F. & Hauskrecht, M. Obtaining Well Calibrated
       Probabilities Using Bayesian Binning. Proc Conf AAAI Artif Intell 2015,
       2901-2907 (2015).
  """

  _setattr_tracking = False  # Automatic tracking breaks some unit tests

  def __init__(self, num_bins=15, name=None, dtype=None):
    """Constructs an expected calibration error metric.

    Args:
      num_bins: Number of bins to maintain over the interval [0, 1].
      name: Name of this metric.
      dtype: Data type.
    """
    super().__init__(name, dtype)
    self.num_bins = num_bins

    self.correct_sums = self.add_weight(
        "correct_sums", shape=(num_bins,), initializer=tf.zeros_initializer)
    self.prob_sums = self.add_weight(
        "prob_sums", shape=(num_bins,), initializer=tf.zeros_initializer)
    self.counts = self.add_weight(
        "counts", shape=(num_bins,), initializer=tf.zeros_initializer)

  def update_state(self,
                   labels,
                   probabilities,
                   custom_binning_score=None,
                   **kwargs):
    """Updates this metric.

    This will flatten the labels and probabilities, and then compute the ECE
    over all predictions.

    Args:
      labels: Tensor of shape [..., ] of class labels in [0, k-1].
      probabilities: Tensor of shape [..., ], [..., 1] or [..., k] of normalized
        probabilities associated with the True class in the binary case, or with
        each of k classes in the multiclass case.
      custom_binning_score: Tensor of shape [..., ] matching the first dimension
        of probabilities used for binning predictions. If not set, the default
        is to bin by predicted probability. The elements of custom_binning_score
        are expected to all be in [0, 1].
      **kwargs: Other potential keywords, which will be ignored by this method.
    """
    del kwargs  # unused
    labels = tf.convert_to_tensor(labels)
    probabilities = tf.cast(probabilities, self.dtype)

    # Flatten labels and custom_binning_score to [N, ].
    if tf.rank(labels) != 1:
      labels = tf.reshape(labels, [-1])
    if custom_binning_score is not None and tf.rank(custom_binning_score) != 1:
      custom_binning_score = tf.reshape(custom_binning_score, [-1])
    # Flatten probabilities to [N, 1] or [N, k].
    if tf.rank(probabilities) != 2 or (tf.shape(probabilities)[0] !=
                                       tf.shape(labels)[0]):
      probabilities = tf.reshape(probabilities, [tf.shape(labels)[0], -1])
    # Extend any probabilities of shape [N, 1] to shape [N, 2].
    # NOTE: XLA does not allow for different shapes in the branches of a
    # conditional statement. Therefore, explicit indexing is used.
    given_k = tf.shape(probabilities)[-1]
    k = tf.math.maximum(2, given_k)
    probabilities = tf.cond(
        given_k < 2,
        lambda: tf.concat([1. - probabilities, probabilities], axis=-1)[:, -k:],
        lambda: probabilities)

    pred_labels = tf.math.argmax(probabilities, axis=-1)
    pred_probs = tf.math.reduce_max(probabilities, axis=-1)
    correct_preds = tf.math.equal(pred_labels,
                                  tf.cast(labels, pred_labels.dtype))
    correct_preds = tf.cast(correct_preds, self.dtype)

    # Bin by pred_probs if a separate custom_binning_score was not set.
    if custom_binning_score is None:
      custom_binning_score = pred_probs

    bin_indices = tf.histogram_fixed_width_bins(
        custom_binning_score,
        tf.constant([0., 1.], self.dtype),
        nbins=self.num_bins)
    batch_correct_sums = tf.math.unsorted_segment_sum(
        data=tf.cast(correct_preds, self.dtype),
        segment_ids=bin_indices,
        num_segments=self.num_bins)
    batch_prob_sums = tf.math.unsorted_segment_sum(data=pred_probs,
                                                   segment_ids=bin_indices,
                                                   num_segments=self.num_bins)
    batch_counts = tf.math.unsorted_segment_sum(data=tf.ones_like(bin_indices),
                                                segment_ids=bin_indices,
                                                num_segments=self.num_bins)
    batch_counts = tf.cast(batch_counts, self.dtype)
    self.correct_sums.assign_add(batch_correct_sums)
    self.prob_sums.assign_add(batch_prob_sums)
    self.counts.assign_add(batch_counts)

  def result(self):
    """Computes the expected calibration error."""
    non_empty = tf.math.not_equal(self.counts, 0)
    correct_sums = tf.boolean_mask(self.correct_sums, non_empty)
    prob_sums = tf.boolean_mask(self.prob_sums, non_empty)
    counts = tf.boolean_mask(self.counts, non_empty)
    accs = correct_sums / counts
    confs = prob_sums / counts
    total_count = tf.reduce_sum(counts)
    return tf.reduce_sum(counts / total_count * tf.abs(accs - confs))

  def reset_states(self):
    """Resets all of the metric state variables.

    This function is called between epochs/steps,
    when a metric is evaluated during training.
    """
    tf.keras.backend.batch_set_value([(v, [0.,]*self.num_bins) for v in
                                      self.variables])


@metrics_base.registry.register("ece")
class ExpectedCalibrationError(metrics_base.KerasMetric):
  """Expected calibration error."""

  def __init__(
      self,
      dataset_info=None,
      use_dataset_labelset=False,
      num_bins=15):
    metric = _KerasECEMetric(num_bins=num_bins)
    super().__init__(
        dataset_info, metric, "ece", take_argmax=False, one_hot=False,
        use_dataset_labelset=use_dataset_labelset)


class _KerasOracleCollaborativeAccuracyMetric(_KerasECEMetric):
  """Oracle Collaborative Accuracy."""

  def __init__(self, fraction=0.01, num_bins=100, name=None, dtype=None):
    """Constructs an expected collaborative accuracy metric.

    Args:
      fraction: the fraction of total examples to send to moderators.
      num_bins: Number of bins to maintain over the interval [0, 1].
      name: Name of this metric.
      dtype: Data type.
    """
    super(_KerasOracleCollaborativeAccuracyMetric, self).__init__(
        num_bins, name, dtype)
    self.fraction = fraction
    self.collab_correct_sums = self.add_weight(
        "collab_correct_sums",
        shape=(num_bins,),
        initializer=tf.zeros_initializer)

  def result(self):
    """Computes the expected calibration error."""
    num_total_example = tf.reduce_sum(self.counts)
    num_oracle_examples = tf.cast(
        int(num_total_example * self.fraction), self.dtype)
    # TODO(lzi): compute the expected number of accurate predictions
    collab_correct_sums = []
    num_oracle_examples_so_far = 0.0
    for i in range(self.num_bins):
      cur_bin_counts = self.counts[i]
      cur_bin_num_correct = self.correct_sums[i]
      if num_oracle_examples_so_far + cur_bin_counts <= num_oracle_examples:
        # Send all examples in the current bin to the oracle.
        cur_bin_num_correct = cur_bin_counts
        num_oracle_examples_so_far += cur_bin_num_correct
      elif num_oracle_examples_so_far < num_oracle_examples:
        # Send num_correct_oracle examples in the current bin to oracle,
        # and have model to predict the rest.
        cur_bin_accuracy = cur_bin_num_correct / cur_bin_counts
        num_correct_oracle = tf.cast(
            num_oracle_examples - num_oracle_examples_so_far, self.dtype)
        num_correct_model = (cur_bin_counts -
                             num_correct_oracle) * cur_bin_accuracy
        cur_bin_num_correct = num_correct_oracle + num_correct_model
        num_oracle_examples_so_far = num_oracle_examples

      collab_correct_sums.append(cur_bin_num_correct)

    self.collab_correct_sums = tf.stack(collab_correct_sums)

    non_empty = tf.math.not_equal(self.counts, 0)
    counts = tf.boolean_mask(self.counts, non_empty)
    collab_correct_sums = tf.boolean_mask(self.collab_correct_sums, non_empty)

    return tf.reduce_sum(collab_correct_sums) / tf.reduce_sum(counts)


@metrics_base.registry.register("oracle_collaborative")
class OracleCollaborativeAccuracy(metrics_base.KerasMetric):
  """Oracle Collaborative Accuracy measures for probabilistic predictions.

  Oracle Collaborative Accuracy measures the usefulness of model robustness
  scores in facilitating human-computer collaboration (e.g., between a neural
  model and an "oracle" human moderator in moderating online toxic comments).

  The idea is that given a large amount of testing examples, the model will
  first generate predictions for all examples, and then send a certain
  percentage of examples that it is not confident about to the human moderators,
  whom we assume can label those examples correctly.

  The goal of this metric is to understand, under capacity constraints on the
  human moderator (e.g., the model is only allowed to send 0.1% of total
  examples to humans), how well the model can collaborate with the human to
  achieve the highest overall accuracy. In this way, the metric attempts to
  quantify the behavior of the full model-moderator system rather than of the
  model alone.

  A model that collaborates with a human oracle well should not be accurate, but
  also capable of quantifying its robustness well (i.e., its robustness should
  be calibrated such that robustness ≅ model accuracy).
  """

  def __init__(
      self,
      dataset_info=None,
      use_dataset_labelset=False,
      fraction=0.01,
      num_bins=100):
    metric = _KerasOracleCollaborativeAccuracyMetric(
        fraction=fraction, num_bins=num_bins)
    super().__init__(
        dataset_info,
        metric,
        "oracle_collaborative",
        take_argmax=False,
        one_hot=False,
        use_dataset_labelset=use_dataset_labelset)

  @tf.function
  def _add_prediction(self,
                      predictions,
                      label,
                      average_predictions,
                      custom_binning_score=None):
    """Feeds the given label and prediction to the underlying Keras metric.

    Args:
      predictions: The batch of predictions, one for each example in the batch.
      label: The batch of labels, one for each example in the batch.
      average_predictions: If set, when multiple predictions are present for
        a dataset element, they will be averaged before processing.
      custom_binning_score: (Optional) Custom score to use for binning
        predictions, one for each example in the batch. If not specified, the
        default is to bin by predicted probability. The elements of
        custom_binning_score are expected to all be in [0, 1].
    """
    if average_predictions:
      predictions = tf.reduce_mean(predictions, axis=0, keepdims=True)
    if self._one_hot:
      label = tf.one_hot(label, self._num_classes)
    if self._take_argmax:
      self._metric.update_state(
          label,
          tf.argmax(predictions, axis=-1),
          custom_binning_score=custom_binning_score)
    else:
      self._metric.update_state(
          label, predictions, custom_binning_score=custom_binning_score)

  def add_predictions(self, model_predictions: types.ModelPredictions,
                      **metadata) -> None:
    try:
      element_id = int(metadata["element_id"])
      if element_id in self._ids_seen:
        raise ValueError(
            "KerasMetric does not support reporting the same id multiple "
            f"times, but you added element id {element_id} twice.")
      else:
        self._ids_seen.add(element_id)
    except KeyError:
      pass

    stacked_predictions = np.stack(model_predictions.predictions)
    if "label" not in metadata:
      raise ValueError("KerasMetric expects a `label` in the metadata."
                       f"Available fields are: {metadata.keys()!r}")
    custom_binning_score = metadata.get("custom_binning_score")

    if self._use_dataset_labelset:
      # pylint: disable=protected-access
      predictions, label = metrics_base._map_labelset(stacked_predictions,
                                                      metadata["label"],
                                                      self._appearing_classes)
      # pylint: enable=protected-access
      self._add_prediction(
          predictions,
          label,
          self._average_predictions,
          custom_binning_score=custom_binning_score)
    else:
      self._add_prediction(
          stacked_predictions,
          metadata["label"],
          self._average_predictions,
          custom_binning_score=custom_binning_score)

  def add_batch(self, model_predictions,
                **metadata: Optional[Dict[str, Any]]) -> None:
    """Adds a batch of predictions for a batch of examples.

    Args:
      model_predictions: The batch of predictions, one for each example in the
        batch.
      **metadata: The batch metadata, possibly including `labels` which is the
        batch of labels and `custom_binning_score` which is the batch of binning
        scores (overriding the default behavior of binning by predicted
        probability), one for each example in the batch.
    """
    # Note that even though the labels are really over a batch of predictions,
    # we use the kwarg "label" to be consistent with the other functions that
    # use the singular name.
    self._average_predictions = False
    label = metadata["label"]
    custom_binning_score = metadata.get("custom_binning_score")
    if self._use_dataset_labelset:
      model_predictions = tf.gather(
          model_predictions, self._appearing_classes, axis=-1)
      model_predictions /= tf.math.reduce_sum(
          model_predictions, axis=-1, keepdims=True)
      label = tf.convert_to_tensor(
          [self._appearing_classes.index(x) for x in label])
    self._add_prediction(
        predictions=model_predictions,
        label=label,
        average_predictions=False,
        custom_binning_score=custom_binning_score)


@metrics_base.registry.register("nll")
class NegativeLogLikelihood(metrics_base.KerasMetric):
  r"""Multi-class negative log likelihood.

  If the true label is k, while the predicted vector of probabilities is
  [p_1, ..., p_K], then the negative log likelihood is -log(p_k).
  """

  def __init__(self, dataset_info=None, use_dataset_labelset=False):
    metric = tf.keras.metrics.SparseCategoricalCrossentropy()
    super().__init__(
        dataset_info, metric, "nll", take_argmax=False, one_hot=False,
        use_dataset_labelset=use_dataset_labelset)


@metrics_base.registry.register("brier")
class Brier(metrics_base.KerasMetric):
  r"""Brier score.

  If the true label is k, while the predicted vector of probabilities is
  [y_1, ..., y_n], then the Brier score is equal to

    \sum_{i != k} y_i^2 + (y_k - 1)^2.
  """

  def __init__(self, dataset_info=None, use_dataset_labelset=False):
    metric = tf.keras.metrics.MeanSquaredError()
    super().__init__(
        dataset_info, metric, "brier", take_argmax=False, one_hot=True,
        use_dataset_labelset=use_dataset_labelset)

  def _add_prediction(self, predictions, label, average_predictions):
    """Feeds the given label and prediction to the underlying Keras metric."""
    if self._num_classes is None:
      self._num_classes = predictions.shape[-1]
    super()._add_prediction(predictions, label, average_predictions)

  def result(self):
    return {"brier": self._num_classes * float(self._metric.result())}


def _get_adaptive_bins(predictions, num_bins):
  """Returns upper edges for binning an equal number of datapoints per bin."""
  predictions = np.asarray(predictions).reshape(-1)
  if np.size(predictions) == 0:
    bin_upper_bounds = np.linspace(0, 1, num_bins + 1)[1:]
  else:
    edge_indices = np.linspace(
        0, np.size(predictions), num_bins, endpoint=False)

    # Round into integers for indexing. If num_bins does not evenly divide
    # len(predictions), this means that bin sizes will alternate between SIZE
    # and SIZE+1.
    edge_indices = np.round(edge_indices).astype(int)

    # If there are many more bins than data points, some indices will be
    # out-of-bounds by one. Set them to be within bounds:
    edge_indices = np.minimum(edge_indices, np.size(predictions) - 1)

    # Obtain the edge values:
    edges = np.sort(predictions)[edge_indices]

    # Following the convention of numpy.digitize, we do not include the leftmost
    # edge (i.e. return the upper bin edges):
    bin_upper_bounds = np.concatenate((edges[1:], [1.]))

  assert len(bin_upper_bounds) == num_bins and bin_upper_bounds[-1] == 1
  return bin_upper_bounds


def _binary_converter(probs):
  """Converts a binary probability vector into a matrix."""
  return np.array([[1 - p, p] for p in probs])


def _one_hot_encode(labels, num_classes=None):
  """One hot encoder for turning a vector of labels into a OHE matrix."""
  if num_classes is None:
    num_classes = len(np.unique(labels))
  return np.eye(num_classes)[labels]


def _is_monotonic(n_bins, bin_assign, labels):
  """Check if the label means in the bins are monotone.

  Args:
    n_bins: number of bins
    bin_assign: array/list of bin indices (int) assigning each example to bin.
    labels: array/list of class labels for each example in probs

  Returns:
    True if the provided bin_assign is monotonic.
  """
  bin_assign = np.array(bin_assign)
  last_ym = -1000
  for i in range(n_bins):
    cur = bin_assign == i
    if any(cur):
      ym = np.mean(labels[cur])
      if ym < last_ym:  # Determine if the predictions are monotonic.
        return False
      last_ym = ym
  return True


def _em_monotonic_sweep(probs, labels):
  """Compute bin assignments equal mass binning scheme."""
  probs = np.squeeze(probs)
  labels = np.squeeze(labels)
  probs = probs if probs.ndim > 0 else np.array([probs])
  labels = labels if labels.ndim > 0 else np.array([labels])

  sort_ix = np.argsort(probs)
  n_examples = len(probs)
  bin_assign = np.zeros((n_examples), dtype=int)

  prev_bin_assign = np.zeros((n_examples), dtype=int)
  for n_bins in range(2, n_examples):
    bin_assign[sort_ix] = np.minimum(
        n_bins - 1, np.floor(
            (np.arange(n_examples) / n_examples) * n_bins)).astype(int)
    if not _is_monotonic(n_bins, bin_assign, labels):
      return prev_bin_assign
    prev_bin_assign = np.copy(bin_assign)
  return bin_assign


def _ew_monotonic_sweep(probs, labels):
  """Monotonic bin sweep using equal width binning scheme."""
  n_examples = len(probs)
  bin_assign = np.zeros((n_examples), dtype=int)
  prev_bin_assign = np.zeros((n_examples), dtype=int)
  for n_bins in range(2, n_examples):
    bin_assign = np.minimum(n_bins - 1, np.floor(probs * n_bins)).astype(int)
    if not _is_monotonic(n_bins, bin_assign, labels):
      return prev_bin_assign
    prev_bin_assign = np.copy(bin_assign)
  return bin_assign


def _get_bin_edges(bin_assign, probs):
  """Convert bin_assign and probs to a set of bin_edges.

  Args:
    bin_assign: array/list of integer bin assignments.
    probs: array/list of corresponding probs to partition
      with bin_edges.

  Returns:
    bin_upper_bounds: array of right-side-edges that partition probs.

  Example:
  probs = [.2, .4, .6, .7, .9, .95]
  bin_assign = [0,0,1,1,2,2]

  bin_edges = get_bin_edges(bin_assign, probs)
  assert bin_edges == [.2, .5, .8, .95]

  Here bin_assign has 3 unique elements; therefore len(bin_edges) == 3+1
  min(bin_edges) == min(probs), and probs[0] == min(probs)
  max(bin_edges) == max(probs), and probs[-1] == max(probs)
  probs should be monotonically non-decreasing

  When an edge splits data, it does so by choosing the middle between
  the largest value in the left-bin and the smallest value in the right-bin.
  """

  bin_assign = np.squeeze(np.array(bin_assign))
  probs = np.squeeze(np.array(probs))
  bin_assign = bin_assign if bin_assign.ndim != 0 else np.array(
      [int(np.array(bin_assign))])
  probs = probs if probs.ndim != 0 else np.array([float(np.array(probs))])

  bin_edges = []
  curr_bin_max = None
  for ci, bin_ind in enumerate(set(bin_assign)):
    curr_bin_vals = probs[bin_assign == bin_ind]
    if len(curr_bin_vals) > 0:  # pylint: disable=g-explicit-length-test
      curr_bin_min = curr_bin_vals.min()
      curr_bin_max = curr_bin_vals.max()
      if ci == 0:
        bin_edges.append(curr_bin_min)
      else:
        bin_edges.append(curr_bin_min * .5 + previous_max * .5)  # pytype: disable=name-error
      previous_max = curr_bin_max
  if curr_bin_max is not None:
    bin_edges.append(curr_bin_max)

  # Validation relationships:
  if len(probs) > 0:  # pylint: disable=g-explicit-length-test
    assert bin_edges[-1] == max(bin_edges) == max(probs)
    assert bin_edges[0] == min(bin_edges) == min(probs)

  bin_upper_bounds = bin_edges[1:]
  return bin_upper_bounds


class _GeneralCalibrationErrorMetric:
  """Implements the space of calibration errors, General Calibration Error.

  For documentation of the parameters, see GeneralCalibrationError.
  """

  def __init__(self,
               binning_scheme,
               max_prob,
               class_conditional,
               norm,
               num_bins=30,
               threshold=0.0,
               datapoints_per_bin=None,
               distribution=None):
    self.binning_scheme = binning_scheme
    self.max_prob = max_prob
    self.class_conditional = class_conditional
    self.norm = norm
    self.num_bins = num_bins
    self.threshold = threshold
    self.datapoints_per_bin = datapoints_per_bin
    self.distribution = distribution
    self.accuracies = None
    self.confidences = None
    self.calibration_error = None
    self.calibration_errors = None

  def _get_mon_sweep_bins(self, probs, labels):
    """Adapter function to delegate bin_assign to the appropriate sweep method.

    Args:
      probs: array/list of corresponding probs to partition
        with bin_edges.
      labels: array/list of class labels for each example in probs
    Returns:
      Array of edges that partition the probabilities.
    """
    assert probs.ndim == 1
    assert labels.ndim == 1
    probs = probs[:, None]
    labels = labels[:, None]

    if self.binning_scheme == "adaptive":
      bin_assign = _em_monotonic_sweep(probs, labels)
    elif self.binning_scheme == "even":
      bin_assign = _ew_monotonic_sweep(probs, labels)
    else:
      raise NotImplementedError

    bin_edges = _get_bin_edges(bin_assign, probs)
    return bin_edges

  def _get_upper_bounds(self, probs_slice, labels):
    """Delegate construction of bin_upper_bounds to appropriate case-handler."""

    if self.binning_scheme == "adaptive" and self.num_bins is not None:
      bin_upper_bounds = _get_adaptive_bins(probs_slice, self.num_bins)
    elif self.binning_scheme == "adaptive" and self.num_bins is None:
      bin_upper_bounds = self._get_mon_sweep_bins(probs_slice, labels)
    elif self.binning_scheme == "even" and self.num_bins is None:
      bin_upper_bounds = self._get_mon_sweep_bins(probs_slice, labels)
    elif self.binning_scheme == "even" and self.num_bins is not None:
      bin_upper_bounds = np.histogram_bin_edges([],
                                                bins=self.num_bins,
                                                range=(0.0, 1.0))[1:]
    else:
      raise NotImplementedError(
          f"Condition not implemented: binning_scheme:{self.binning_scheme}, "
          f"num_bins:{self.num_bins}"
      )

    return bin_upper_bounds

  def _get_calibration_error(self, probs, labels, bin_upper_bounds):
    """Given a binning scheme, returns sum weighted calibration error."""
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    if np.size(probs) == 0:
      return 0.

    bin_indices = np.digitize(probs, bin_upper_bounds)
    sums = np.bincount(bin_indices, weights=probs, minlength=self.num_bins)
    sums = sums.astype(np.float64)  # In case all probs are 0/1.
    counts = np.bincount(bin_indices, minlength=self.num_bins)
    counts = counts + np.finfo(sums.dtype).eps  # Avoid division by zero.
    self.confidences = sums / counts
    self.accuracies = np.bincount(
        bin_indices, weights=labels, minlength=self.num_bins) / counts

    self.calibration_errors = self.accuracies - self.confidences

    if self.norm == "l1":
      calibration_errors_normed = self.calibration_errors
    elif self.norm == "l2":
      calibration_errors_normed = np.square(self.calibration_errors)
    else:
      raise ValueError(f"Unknown norm: {self.norm}")

    weighting = counts / float(len(probs.flatten()))
    weighted_calibration_error = calibration_errors_normed * weighting

    return np.sum(np.abs(weighted_calibration_error))

  def update_state(self, labels, probs):
    """Updates the value of the General Calibration Error."""

    probs = np.array(probs)
    labels = np.array(labels)
    if probs.ndim == 2:

      num_classes = probs.shape[1]
      if num_classes == 1:
        probs = probs[:, 0]
        probs = _binary_converter(probs)
        num_classes = 2
    elif probs.ndim == 1:
      # Cover binary case
      probs = _binary_converter(probs)
      num_classes = 2
    else:
      raise ValueError("Probs must have 1 or 2 dimensions.")

    # Convert the labels vector into a one-hot-encoded matrix.

    labels_matrix = _one_hot_encode(labels, probs.shape[1])

    if self.datapoints_per_bin is not None:
      self.num_bins = int(len(probs) / self.datapoints_per_bin)
      if self.binning_scheme != "adaptive":
        raise ValueError(
            "To set datapoints_per_bin, binning_scheme must be 'adaptive'.")

    # When class_conditional is False, different classes are conflated.
    if not self.class_conditional:
      if self.max_prob:
        labels_matrix = labels_matrix[range(len(probs)),
                                      np.argmax(probs, axis=1)]
        probs = probs[range(len(probs)), np.argmax(probs, axis=1)]
      labels = np.squeeze(labels_matrix[probs > self.threshold])
      probs_slice = np.squeeze(probs[probs > self.threshold])
      bin_upper_bounds = self._get_upper_bounds(probs_slice, labels)
      calibration_error = self._get_calibration_error(probs_slice, labels,
                                                      bin_upper_bounds)

    # If class_conditional is true, predictions from different classes are
    # binned separately.
    else:
      # Initialize list for class calibration errors.
      class_calibration_error_list = []
      for j in range(num_classes):
        if not self.max_prob:
          probs_slice = probs[:, j]
          labels = labels_matrix[:, j]
          labels = labels[probs_slice > self.threshold]
          probs_slice = probs_slice[probs_slice > self.threshold]
          bin_upper_bounds = self._get_upper_bounds(probs_slice, labels)

          calibration_error = self._get_calibration_error(
              probs_slice, labels, bin_upper_bounds)
          class_calibration_error_list.append(calibration_error / num_classes)
        else:
          # In the case where we use all datapoints,
          # max label has to be applied before class splitting.
          labels = labels_matrix[np.argmax(probs, axis=1) == j][:, j]
          probs_slice = probs[np.argmax(probs, axis=1) == j][:, j]
          labels = labels[probs_slice > self.threshold]
          probs_slice = probs_slice[probs_slice > self.threshold]
          bin_upper_bounds = self._get_upper_bounds(probs_slice, labels)
          calibration_error = self._get_calibration_error(
              probs_slice, labels, bin_upper_bounds)
          class_calibration_error_list.append(calibration_error / num_classes)
      calibration_error = np.sum(class_calibration_error_list)

    if self.norm == "l2":
      calibration_error = np.sqrt(calibration_error)

    self.calibration_error = calibration_error

  def result(self):
    return self.calibration_error

  def reset_states(self):
    self.calibration_error = None


@metrics_base.registry.register("gce")
class GeneralCalibrationError(metrics_base.FullBatchMetric):
  """Implements a large set of of calibration errors.

  This implementation of General Calibration Error can be class-conditional,
  adaptively binned, thresholded, focus on the maximum or top labels, and use
  the l1 or l2 norm. Can function as ECE, SCE, RMSCE, and more. For
  definitions of most of these terms, see [1].

  The metric returns a dict with keys:
    * "gce":  General Calibration Error. This is returned for all
        recalibration_method values, including None.
    * "beta": Optimal beta scaling, returned only for the temperature_scaling
        recalibration method

  Note that we also implement the following metrics by specializing this class
  and fixing some of its parameters:

  Static Calibration Error [1], registered under name "sce":
    binning_scheme="even"
    class_conditional=False
    max_prob=False
    norm="l1"

  Root Mean Squared Calibration Error [3], registered under "rmsce":
    binning_scheme="adaptive"
    class_conditional=False
    max_prob=True
    norm="l2"
    datapoints_per_bin=100

  Adaptive Calibration Error [1], registered under "ace":
    binning_scheme="adaptive"
    class_conditional=True
    max_prob=False
    norm="l1"

  Thresholded Adaptive Calibration Error [1], registered under "tace":
    binning_scheme="adaptive"
    class_conditional=True
    max_prob=False
    norm="l1"
    threshold=0.01

  Monotonic Sweep Calibration Error [4], registered under "msce":
    binning_scheme="adaptive"
    class_conditional=False
    max_prob=True
    norm="l1"
    num_bins=None

  ### References

  [1] Nixon, Jeremy, Michael W. Dusenberry, Linchuan Zhang, Ghassen Jerfel,
  and Dustin Tran. "Measuring Calibration in Deep Learning." In Proceedings of
  the IEEE Conference on Computer Vision and Pattern Recognition Workshops,
  pp. 38-41. 2019.
  https://arxiv.org/abs/1904.01685

  [2] Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht.
  "Obtaining well calibrated probabilities using bayesian binning."
  Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4410090/

  [3] Khanh Nguyen and Brendan O’Connor.
  "Posterior calibration and exploratory analysis for natural language
  processing models."  Empirical Methods in Natural Language Processing. 2015.
  https://arxiv.org/pdf/1508.05154.pdf

  [4] Rebecca Roelofs, Nicholas Cain, Jonathon Shlens, Michael C. Mozer
  "Mitigating bias in calibration error estimation."
  https://arxiv.org/pdf/2012.08668.pdf
  """

  def __init__(
      self,
      dataset_info: datasets_base.DatasetInfo,
      binning_scheme: str,
      max_prob: bool,
      class_conditional: bool,
      norm: str,
      num_bins: Optional[int],
      threshold: float,
      datapoints_per_bin: Optional[int] = None,
      fit_on_percent: float = 100.0,
      recalibration_method: Optional[str] = None,
      seed: Optional[int] = None,
      use_dataset_labelset: bool = False,
      pickle_path: str = DEFAULT_PICKLE_PATH,
  ):
    """Initializes the GCE metric.

    If neither num_bins nor datapoints_per_bin are set, the bins are set using
    the monotone strategy in [4].

    Args:
      dataset_info: A datasets.DatasetInfo object.
      binning_scheme: Either "even" (for even spacing) or "adaptive"
        (for an equal number of datapoints in each bin).
      max_prob: "True" to measure calibration only on the maximum
        prediction for each datapoint, "False" to look at all predictions.
      class_conditional: "False" for the case where predictions from
        different classes are binned together, "True" for binned separately.
      norm: Apply "l1" or "l2" norm to the calibration error.
      num_bins: Number of bins of confidence scores to use.
      threshold: Ignore predictions below this value.
      datapoints_per_bin: When using an adaptive binning scheme, this determines
        the number of datapoints in each bin.
      fit_on_percent: Percentage of data used to fit recalibration function.
      recalibration_method: Takes values "temperature_scaling",
        "isotonic_regression" and None.
      seed: Randomness seed used for data shuffling before recalibration split.
      use_dataset_labelset: If set, and the given dataset has only a subset of
        the clases the model produces, the classes that are not in the dataset
        will be removed and the others scaled to sum up to one.
      pickle_path: Path to where the isotonic regression model will be
        pickled. The model encompasses many parameters, all of which are
        generated and loaded by separate processes.
    """

    self._ids_seen = set()
    self._predictions = []
    self._labels = []
    self._eval_predictions = []
    self._eval_labels = []
    self._fit_predictions = []
    self._fit_labels = []

    self._binning_scheme = binning_scheme
    self._max_prob = max_prob
    self._class_conditional = class_conditional
    self._norm = norm
    self._num_bins = num_bins
    self._threshold = threshold
    self._datapoints_per_bin = datapoints_per_bin
    self._fit_on_percent = fit_on_percent
    self._seed = seed
    self._pickle_path = pickle_path
    if not 0 <= fit_on_percent <= 100:
      raise ValueError(f"Argument fit_on_percent={fit_on_percent} is not within"
                       " expected range [0,100].")
    self._recalibration_method = recalibration_method
    if fit_on_percent == 100.0 and recalibration_method is not None:
      warnings.warn("Recalibration without data split: You are both fitting and"
                    " rescaling on the entire data set (method: "
                    f"{recalibration_method}). Set 'fit_on_percent'<100 or "
                    "recalibration_method=None.")
    if fit_on_percent == 0.0 and recalibration_method is not None:
      warnings.warn("No recalibration without fitting data: You selected the "
                    f"recalibration method {recalibration_method} and specified"
                    f" fitting on {fit_on_percent} percent of the data. "
                    " Recalibration is skipped. Set 'fit_on_percent'>0 to "
                    "recalibrate data with selected method.")

    super().__init__(dataset_info, use_dataset_labelset=use_dataset_labelset)

  def result(self) -> Dict[str, float]:
    self.shuffle_and_split_data()
    if self._recalibration_method == "temperature_scaling":
      beta = _temperature_scaling(self._fit_predictions, self._fit_labels)
      pred_dtype = type(self._eval_predictions[0][0])
      logits = np.log(self._eval_predictions + np.finfo(pred_dtype).eps)
      self._eval_predictions = tf.nn.softmax(beta * logits, axis=-1).numpy()
    elif self._recalibration_method == "isotonic_regression":
      ir = IsotonicRegression(pickle_path=self._pickle_path)
      ir.fit(self._fit_predictions, self._fit_labels)
      self._eval_predictions = ir.scale(self._eval_predictions)
    elif self._recalibration_method is not None:
      raise ValueError("You added an unknown recalibration method: "
                       f"{self._recalibration_method}. Supported options: "
                       "'temperature_scaling', 'isotonic_regression', None .")
    m = _GeneralCalibrationErrorMetric(
        binning_scheme=self._binning_scheme,
        max_prob=self._max_prob,
        class_conditional=self._class_conditional,
        norm=self._norm,
        num_bins=self._num_bins,
        threshold=self._threshold,
        datapoints_per_bin=self._datapoints_per_bin,
    )
    m.update_state(np.asarray(self._eval_labels),
                   np.asarray(self._eval_predictions))
    if self._recalibration_method == "temperature_scaling":
      return {"gce": m.result(), "beta": beta}
    return {"gce": m.result()}

  def shuffle_and_split_data(self) -> None:
    n_labels = len(self._labels)
    number_of_fit_examples = round(self._fit_on_percent*0.01*n_labels)
    labels = np.asarray(self._labels)
    predictions = np.asarray(self._predictions)
    if number_of_fit_examples == n_labels:
      # No shuffling, no data split.
      # Fit and evaluate on the (same) complete data set.
      self._eval_predictions = predictions
      self._eval_labels = labels
      self._fit_predictions = predictions
      self._fit_labels = labels
    else:
      # Shuffle ordered pair (labels, predictions) by using the permutation
      # method of the random number generator.
      # After updating to numpy 1.17 : use
      # rng = np.random.default_rng(seed=self._seed)
      # perm = rng.permutation(len(lbls))
      perm = np.random.RandomState(seed=self._seed).permutation(n_labels)
      labels, predictions = labels[perm], predictions[perm]
      self._eval_predictions = predictions[number_of_fit_examples:]
      self._eval_labels = labels[number_of_fit_examples:]
      self._fit_predictions = predictions[:number_of_fit_examples]
      self._fit_labels = labels[:number_of_fit_examples]


# TODO(mjlm): Consider moving non-metric code to separate file:
def _temperature_scaling(predictions, labels):
  """Fits temperature scaling on predictions to improve calibration.

  Based on Guo et al., 2017 (https://arxiv.org/abs/1706.04599).

  Args:
    predictions: Array of predictions (post-softmax) with shape [num_examples,
      num_classes].
    labels: Array of labels for 'predictions' with shape [num_fit_examples].

  Returns:
    Optimal temperature scaling parameter beta.
  """

  # Taking the log will recover the model output logits up to an additive
  # constant, which will not affect the results:
  logits = np.log(predictions + np.finfo(type(predictions[0][0])).eps)

  with tf.device("job:localhost"):
    def objective(beta):
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits * beta, labels=labels)
      return tf.reduce_sum(cross_entropy)

    optim_results = tfp.optimizer.lbfgs_minimize(
        tf.function(lambda beta: tfp.math.value_and_gradient(objective, beta)),
        initial_position=[1.0])

    if not optim_results.converged:
      logging.warn("LBFG-S did not converge during temperature scaling!")

    return optim_results.position[0]


@metrics_base.registry.register("temperature_scaling")
class TemperatureScaling(metrics_base.FullBatchMetric):
  """Identifies the optimal temperature scaling parameter.

  See Guo et al., 2017 (https://arxiv.org/abs/1706.04599) for details on
  temperature scaling.
  """

  def result(self) -> Dict[str, float]:
    labels = np.asarray(self._labels)
    predictions = np.asarray(self._predictions)
    assert len(labels) == len(predictions), "Labels/predictions don't match."
    beta = _temperature_scaling(predictions=predictions, labels=labels)
    return {"beta": float(beta)}


@metrics_base.registry.register("isotonic_scaling")
class IsotonicRegression(metrics_base.FullBatchMetric):
  """Fits an isotonic regression model on predictions to improve calibration.

  Stores the model as a pickled sklearn object at a given pickle_path.
  Based on Zadrozny & Elkan, 2002, (https://openreview.net/forum?id=S14ZlV-ObB).
  """

  def __init__(self,
               dataset_info=None,
               pickle_path="",
               use_dataset_labelset=False
               ):
    """Initializes an IsotonicRegression instance.

    Args:
     dataset_info: dataset info, from metrics_base.FullBatchMetric
     pickle_path: path to where the rescaling parameters are stored
       to after fitting and fetched from before rescaling.  Note that
       `self._pickle_class_paths` is added before fitting as a list with
       index:detection class and value:path to location. Having a different path
       for each class is necessary as long as the rescaling parameters are
       calculated independently for each class.
     use_dataset_labelset: Boolean values, from metrics_base.FullBatchMetric
    """
    if not pickle_path:
      self._pickle_path = DEFAULT_PICKLE_PATH
    else:
      self._pickle_path = pickle_path
    super().__init__(dataset_info)

  def _get_pickle_paths(self, number_of_classes: int) ->  Union[List[str], str]:
    """Returns paths to where rescaling params are written are stored in a list.

    Args:
      number_of_classes: number of model classes for detection.
    """
    if number_of_classes > 1:
      pickle_class_paths = []
      for class_i in range(number_of_classes):
        pickle_class_paths.append(
            os.path.join(self._pickle_path, f"class_{class_i}.pickle"))
    else:
      pickle_class_paths = os.path.join(self._pickle_path, "class_0.pickle")
    return pickle_class_paths

  def fit(self, all_predictions: List[np.ndarray],
          all_labels: List[int]) -> None:
    """Fits an isotonic regression to the input paramaters.

    Zadrozny & Elkan, 2002, (https://openreview.net/forum?id=S14ZlV-ObB).
    Instantiates a sklearn model, fits it to the data and
    writes it to disk by pickling the object. Process is repeated for every
    class independently.

    Args:
      all_predictions: 2D list with model probabilities as [instance][class]
      all_labels: 1D list of class labels as [instance]
    """
    all_predictions = np.asarray(all_predictions)
    all_labels = np.asarray(all_labels)

    if all_predictions.ndim == 1:
      pickle_paths = self._get_pickle_paths(all_predictions.ndim)
      predictions = all_predictions[:].astype(np.float64)
      labels = all_labels.astype(int).astype(np.int32)
      ir = sklearn_ir.IsotonicRegression(out_of_bounds="clip")
      ir.fit(predictions, labels)
      with tf.io.gfile.GFile(pickle_paths, "wb") as handle:
        pickle.dump(ir, handle)
    else:
      number_of_classes = all_predictions.shape[1]
      pickle_paths = self._get_pickle_paths(number_of_classes)
      for class_i in range(number_of_classes):
        predictions = all_predictions[:, class_i].astype(np.float64)
        labels = (all_labels == class_i).astype(int).astype(np.int32)
        ir = sklearn_ir.IsotonicRegression(out_of_bounds="clip")
        ir.fit(predictions, labels)
        with tf.io.gfile.GFile(pickle_paths[class_i],
                               "wb") as handle:
          pickle.dump(ir, handle)

  def scale(self,
            all_predictions: np.ndarray) -> np.ndarray:
    calibrated_predictions = []
    if all_predictions.ndim == 1:
      pickle_paths = self._get_pickle_paths(all_predictions.ndim)
      with tf.io.gfile.GFile(pickle_paths, "rb") as handle:
        ir = pickle.load(handle)
      predictions = all_predictions.astype(np.float64)
      calibrated_predictions.append(ir.transform(predictions))
    else:
      number_of_classes = all_predictions.shape[1]
      pickle_paths = self._get_pickle_paths(number_of_classes)
      for class_i in range(number_of_classes):
        with tf.io.gfile.GFile(pickle_paths[class_i],
                               "rb") as handle:
          ir = pickle.load(handle)
        # rescale predictions for each class
        predictions = all_predictions[:, class_i].astype(np.float64)
        calibrated_predictions.append(ir.transform(predictions))
    all_calibrated_predictions = np.stack(calibrated_predictions, axis=1)
    # Normalize the scaled probabilities for each instance across all classes
    all_calibrated_predictions /= np.sum(all_calibrated_predictions,
                                         axis=1,
                                         keepdims=True)
    return all_calibrated_predictions

  def result(self) -> Dict[str, float]:
    labels = np.asarray(self._labels)
    predictions = np.asarray(self._predictions)
    assert len(labels) == len(predictions), "Labels/predictions don't match."
    self.fit(predictions, labels)
    return {"dummy": 0.0}


@metrics_base.registry.register("crps")
class CRPSSCore(metrics_base.FullBatchMetric):
  r"""Computes the Continuous Ranked Probability Score (CRPS).

  The Continuous Ranked Probability Score is a [proper scoring rule][1] for
  assessing the probabilistic predictions of a model against a realized value.
  The CRPS is

  \\(\textrm{CRPS}(F,y) = \int_{-\inf}^{\inf} (F(z) - 1_{z \geq y})^2 dz.\\)

  Here \\(F\\) is the cumulative distribution function of the model predictive
  distribution and \\(y)\\ is the realized ground truth value.

  The CRPS can be used as a loss function for training an implicit model for
  probabilistic regression.  It can also be used to assess the predictive
  performance of a probabilistic regression model.

  In this implementation we use an equivalent representation of the CRPS,

  \\(\textrm{CRPS}(F,y) = E_{z~F}[|z-y|] - (1/2) E_{z,z'~F}[|z-z'|].\\)

  This equivalent representation has an unbiased sample estimate and our
  implementation of the CRPS has a complexity is O(n*m).

  It is expected that the predictions (for each point) are of shape (m,) holding
  m samples o the model predictive p(y|x_i). It is further expected that the
  metadat has a field `label` holding the real-valued target.

  The computed result has a single key "crps", holding the average CRPS.

  #### References
  [1]: Tilmann Gneiting, Adrian E. Raftery.
       Strictly Proper Scoring Rules, Prediction, and Estimation.
       Journal of the American Statistical Association, Vol. 102, 2007.
       https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
  """

  def result(self):
    pairwise_diff = tf.roll(self._predictions, 1, axis=1) - self._predictions
    predictive_diff = tf.abs(pairwise_diff)
    estimated_dist_pairwise = tf.reduce_mean(input_tensor=predictive_diff,
                                             axis=1)
    labels = tf.expand_dims(self._labels, 1)
    dist_realization = tf.reduce_mean(tf.abs(self._predictions-labels), axis=1)

    crps = dist_realization - 0.5*estimated_dist_pairwise

    return {"crps": crps}


@tf.function
def _compute_brier_decomposition(probabilities, labels):
  """Computes the Brier decomposition."""
  _, nlabels = probabilities.shape  # Implicit rank check.

  # Compute pbar, the average distribution
  pred_class = tf.argmax(probabilities, axis=1, output_type=tf.int32)
  confusion_matrix = tf.math.confusion_matrix(pred_class, labels, nlabels,
                                              dtype=tf.float32)
  dist_weights = tf.reduce_sum(confusion_matrix, axis=1)
  dist_weights /= tf.reduce_sum(dist_weights)
  pbar = tf.reduce_sum(confusion_matrix, axis=0)
  pbar /= tf.reduce_sum(pbar)

  # dist_mean[k,:] contains the empirical distribution for the set M_k
  # Some outcomes may not realize, corresponding to dist_weights[k] = 0
  dist_mean = confusion_matrix / tf.expand_dims(
      tf.reduce_sum(confusion_matrix, axis=1) + 1.0e-7, 1)

  # Uncertainty: quadratic entropy of the average label distribution
  uncertainty = -tf.reduce_sum(tf.square(pbar))

  # Resolution: expected quadratic divergence of predictive to mean
  resolution = tf.square(tf.expand_dims(pbar, 1) - dist_mean)
  resolution = tf.reduce_sum(dist_weights * tf.reduce_sum(resolution, axis=1))

  # Reliability: expected quadratic divergence of predictive to true
  prob_true = tf.gather(dist_mean, pred_class, axis=0)
  reliability = tf.reduce_sum(tf.square(prob_true - probabilities), axis=1)
  reliability = tf.reduce_mean(reliability)

  return {"uncertainty": uncertainty,
          "resolution": resolution,
          "reliability": reliability}


@metrics_base.registry.register("brier_decomposition")
class BrierDecomposition(metrics_base.FullBatchMetric):
  r"""Decompose the Brier score into uncertainty, resolution, and reliability.

  [Proper scoring rules][1] measure the quality of probabilistic predictions;
  any proper scoring rule admits a [unique decomposition][2] as
  `Score = Uncertainty - Resolution + Reliability`, where:

  * `Uncertainty`, is a generalized entropy of the average predictive
    distribution; it can both be positive or negative.
  * `Resolution`, is a generalized variance of individual predictive
    distributions; it is always non-negative.  Difference in predictions reveal
    information, that is why a larger resolution improves the predictive score.
  * `Reliability`, a measure of calibration of predictions against the true
    frequency of events.  It is always non-negative and a lower value here
    indicates better calibration.

  This method estimates the above decomposition for the case of the Brier
  scoring rule for discrete outcomes.  For this, we need to discretize the space
  of probability distributions; we choose a simple partition of the space into
  `nlabels` events: given a distribution `p` over `nlabels` outcomes, the index
  `k` for which `p_k > p_i` for all `i != k` determines the discretization
  outcome; that is, `p in M_k`, where `M_k` is the set of all distributions for
  which `p_k` is the largest value among all probabilities.

  The estimation error of each component is O(k/n), where n is the number
  of instances and k is the number of labels.  There may be an error of this
  order when compared to `brier_score`.

  The computed result has three string keys "uncertainty", "resolution" and
  "reliability", holding the corresponding terms of the decomposition.

  #### References
  [1]: Tilmann Gneiting, Adrian E. Raftery.
       Strictly Proper Scoring Rules, Prediction, and Estimation.
       Journal of the American Statistical Association, Vol. 102, 2007.
       https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
  [2]: Jochen Broecker.  Reliability, sufficiency, and the decomposition of
       proper scores.
       Quarterly Journal of the Royal Meteorological Society, Vol. 135, 2009.
       https://rmets.onlinelibrary.wiley.com/doi/epdf/10.1002/qj.456
  """

  def result(self):
    return _compute_brier_decomposition(np.asarray(self._predictions),
                                        np.asarray(self._labels))


@metrics_base.registry.register("semiparametric_ce")
class SemiParametricCalibrationError(metrics_base.FullBatchMetric):
  """Semiparametric calibration.

  This meric estimates L2 calibration error using a semi-parametric method that
  reduces the bias of binning estimates of accuracy conditional on confidence.

  The result dictionary has a single key "ce" holding the calibration error.
  """

  def __init__(self, dataset_info=None, folds=5, weight_trunc=0.05,
               weights="constant", bootstrap_size=500, orthogonal=False,
               normalize=False, smoothing="kernel", hyperparam_attempts=50,
               hyperparam_range=None,
               fold_generator=None):  # pylint: disable=g-doc-args
    """Initializes the object.

    Args:
      dataset_info: The dataset info describing the dataset.
      folds: Used for cross validation of hyperparameter (smoothness).
      weight_trunc: Truncates relative L2 error weights to avoid variance
        blowup.
      weights: Choice of weighted L2 error, either `constant`, `relative`
        (which is `1/max(prediction * (1 - prediction), weight_trunc)`),
        or `chi` which refers to chi-squared weighting, which is sqrt of
        the relative weights.
      bootstrap_size: number of bootstrap samples when computing confidence
        intervals for the L2 error.
      orthogonal: Use full orthogonalized influence function for estimation
        (otherwise it uses the influence function only for standard errors).
      normalize: Normalize the weights, or not. Sometimes, it is helpful to
        interpret the relative weights in an un-normalized way.
      smoothing: To perform smoothing to learn the calibration function,
        either use `spline` or `kernel` smoothing.
      hyperparam_attempts: For cross-validation, how many different values
        in the `hyperparam_range` to try.
      hyperparam_range: Smoothing parameters to try when estimating the
        calibration function. If None is provided, will try to use reasonable
        defaults that worked well in simulation.
      fold_generator: If provided, uses the passed in sklearn fold
        generator. Otherwise, uses a `StratifiedKFold` generator, stratified
        on the labels.
    """
    # choices as well as cross fitting of semiparametric nuisance params.
    super().__init__(dataset_info=dataset_info)
    self.folds = folds
    if fold_generator is None:
      fold_generator = sklearn.model_selection.StratifiedKFold(
          n_splits=folds, shuffle=True, random_state=708)
    self.kf = fold_generator
    self.weight_trunc = weight_trunc
    self.orthogonal = orthogonal
    self.bootstrap_size = bootstrap_size
    self.smoothing = smoothing
    self.normalize = normalize
    self.hyperparam_attempts = hyperparam_attempts
    self.weights = weights

    if hyperparam_range is None:
      # Use reasonable default hyperparam ranges validated in simulation. These
      # scale adaptive estimates made once basic properties of the data are
      # known.
      if smoothing == "kernel":
        hyperparam_range = (0, 0.1)
      if smoothing == "spline":
        hyperparam_range = (0.5, 1.1)

    self.default_hyperparam_range = hyperparam_range

  def _relative_weights(self, probs):
    return 1.0 / np.maximum(np.minimum(probs**2, (1 - probs)**2),
                            self.weight_trunc)

  def _max_power_weights(self, probs):
    return 1.0 / np.maximum(probs * (1 - probs), self.weight_trunc)

  def _constant_weights(self, probs):
    return np.ones(probs.shape[0])

  def weight_function(self, probs):
    if self.weights is None or self.weights == "constant":
      return self._constant_weights(probs)
    elif self.weights == "relative":
      return self._relative_weights(probs)
    elif self.weights == "chi":
      return self._max_power_weights(probs)
    else:
      raise ValueError(f"Unknown weight function {self.weights!r}")

  def _weighted_mean(self, samples, weights):
    if self.normalize:
      return np.mean(samples * weights) / np.mean(weights)
    else:
      return np.mean(samples * weights)

  def _weighted_se(self, samples, weights):
    n = samples.shape[0]
    boot_samps = np.zeros(self.bootstrap_size)
    for b in range(self.bootstrap_size):
      boot_idx = np.random.choice(n, n)
      boot_samps[b] = self._weighted_mean(samples[boot_idx], weights[boot_idx])
    return np.std(boot_samps)

  def _calculate_calibration_error(self, probs, labels, accs):
    """Compute L2 calibration error using semiparametric method."""
    # Given data and nuisance parameter estimates, compute parameter
    # estimate---the L2 calibration error---and it's standard error.
    stat = (probs - labels) * (probs - accs)
    if_ = (probs - accs) * (probs + accs - 2 * labels)

    if self.orthogonal:
      est = self._weighted_mean(if_, self.weight_function(probs))
    else:
      est = self._weighted_mean(stat, self.weight_function(probs))
    se = self._weighted_se(if_, self.weight_function(probs))

    return est, se

  def _calculate_calibration_error_crossfit(self, probs, labels):
    """Compute calib error using best hyperparams for calibration function."""
    if self.smoothing == "spline":
      w = self.weight_function(probs)
      w /= np.mean(w)
      max_val = np.sum(w ** 2 * probs * (1-probs))
      scale_lower, scale_upper = self.default_hyperparam_range
      hyperparam_range = np.linspace(scale_lower * max_val,
                                     scale_upper * max_val,
                                     self.hyperparam_attempts)
    if self.smoothing == "kernel":
      scale_lower, scale_upper = self.default_hyperparam_range
      hyperparam_range = (np.linspace(
          scale_lower * probs.shape[0] + 1, scale_upper * probs.shape[0],
          self.hyperparam_attempts) / (np.max(probs) - np.min(probs)))**2

    return self._calculate_calibration_error(
        probs, labels,
        self._calculate_opt_cross_fit_calibration_function(
            probs, labels, hyperparam_range))

  def _calculate_calibration_function(self, train_probs, train_labels,
                                      test_probs, sigma=1):
    """Compute smoothing estimate of calibration function."""
    # Calibration function is the expected accuracy conditional on the
    # confidence / probabilities outputted by the prediction model. This
    # fits a smoothing model for the accuracy on the train data and gets the
    # model predictions on the test_probs.
    weights = self.weight_function(train_probs)
    weights /= np.mean(weights)
    train_labels -= train_probs
    if self.smoothing == "kernel":
      dists = np.abs(train_probs[np.newaxis, :] - test_probs[:, np.newaxis])
      kernel = np.exp(-sigma * (dists ** 2))
      preds = kernel.dot(train_labels) / kernel.sum(axis=1)
    elif self.smoothing == "spline":
      order = np.argsort(train_probs)
      s = scipy.interpolate.UnivariateSpline(
          train_probs[order], train_labels[order], s=sigma, w=weights)
      preds = s(test_probs)
    else:
      raise Exception(f"Smoothing type {self.smoothing!r} not implemented")
    preds += test_probs
    return preds

  def _calculate_cross_fit_calibration_function(self, probs, labels,
                                                hyperparams):
    """Helper function to estimate the calibration function w/ cross fitting."""
    accs = np.zeros(labels.shape)
    for train_index, test_index in self.kf.split(probs, labels):
      train_probs, test_probs = probs[train_index], probs[test_index]
      train_labels = labels[train_index]
      accs[test_index] = self._calculate_calibration_function(
          train_probs, train_labels, test_probs, hyperparams)
    return accs

  def _choose_opt_calibration_hyperparam(self, probs, labels, hyperparam_range):
    """Gets optimal prediction hyperparam from list hyperparam_range."""
    weights = self.weight_function(probs)
    weights /= np.mean(weights)

    best_error = np.float("inf")
    best_hyperparam = None
    for hyperparam in hyperparam_range:
      accs = self._calculate_cross_fit_calibration_function(
          probs, labels, hyperparam)
      error = np.mean(weights * (accs - labels) ** 2)
      if error < best_error:
        best_error = error
        best_hyperparam = hyperparam
    logging.info("Tried hyperparams: %s", hyperparam_range)
    logging.info("Best hyperparam: %s", best_hyperparam)
    return best_hyperparam

  def _get_undersmoothed_hyperparam(self, probs, labels, hyperparam_range):
    """Adjust optimal hyperparam to work for semiparametric estimation."""
    # The optimal hyperparams for prediction of accuracy with the calibration
    # function are generally more smooth than one wants when plugging them in to
    # a semiparametric estimator. This takes the optimal hyperparams for
    # prediction and fudges them a little bit to undersmooth by just the right
    # amount.
    n = labels.shape[0]
    opt_hyperparam = self._choose_opt_calibration_hyperparam(
        probs, labels, hyperparam_range)
    if self.smoothing == "kernel":
      opt_hyperparam *= n ** 0.08
    else:
      # For now, use hacky adjustment, since it"s not clear how to choose.
      # Should be slightly undersmoothed, but amount should depend on n.
      # In simulation, this works well for n between 1000 and 20000.
      opt_hyperparam *= 0.985
    return opt_hyperparam

  def _calculate_opt_cross_fit_calibration_function(
      self, probs, labels, hyperparam_range):
    """Get best, cross fit calibration function for semiparametric estimator."""
    hyperparam = self._get_undersmoothed_hyperparam(
        probs, labels, hyperparam_range)

    return self._calculate_cross_fit_calibration_function(
        probs, labels, hyperparam)

  def result(self):
    """Low bias estimate of L2 calibration error w/ smoothing, not bins."""
    est, _ = self._calculate_calibration_error_crossfit(
        np.asarray(self._predictions), np.asarray(self._labels),
    )
    return {"ce": np.sqrt(max(est, 0))}


@metrics_base.registry.register("semiparametric_ce_ci")
class SemiParametricCalibrationErrorConfidenceInterval(
    SemiParametricCalibrationError):
  """Semiparametric calibration with a confidence interval.

  This meric estimates L2 calibration error using a semi-parametric method that
  reduces the bias of binning estimates of accuracy conditional on confidence,
  and provides statistically valid confidence intervals for the calibration
  error.

  The result dictionary has a three keys:
     * "ce": holding the calibration error
     * "low": lower bound of the confidence interval
     * "high": upper bound of the confidence interval
  """

  def result(self):
    """Confidence interval for L2 calibration error."""
    # Estimates L2 calibration error using a semi-parametric method that
    # reduces the bias of binning estimates of accuracy conditional on
    # confidence, and provides statistically valid confidence intervals for the
    # calibration error.
    est, se = self._calculate_calibration_error_crossfit(
        np.asarray(self._predictions), np.asarray(self._labels))
    alpha = 0.05
    z_alpha_div_2 = -scipy.stats.norm.ppf(alpha / 2.0)
    return {"low": np.sqrt(max(est - z_alpha_div_2 * se, 0)),
            "ce": np.sqrt(max(est, 0)),
            "high": np.sqrt(max(est + z_alpha_div_2 * se, 0))}


@metrics_base.registry.register("rmsce")
class RootMeanSquaredCalibrationError(GeneralCalibrationError):

  def __init__(self,
               dataset_info: Optional[datasets_base.DatasetInfo] = None,
               num_bins: int = 30,
               **kwargs):
    super().__init__(dataset_info,
                     threshold=0,
                     binning_scheme="adaptive",
                     max_prob=True,
                     class_conditional=False,
                     norm="l2",
                     num_bins=num_bins,
                     **kwargs)


@metrics_base.registry.register("sce")
class StaticCalibrationError(GeneralCalibrationError):

  def __init__(self,
               dataset_info: Optional[datasets_base.DatasetInfo] = None,
               num_bins: int = 30,
               **kwargs):
    super().__init__(dataset_info,
                     threshold=0,
                     binning_scheme="even",
                     max_prob=False,
                     class_conditional=True,
                     norm="l1",
                     num_bins=num_bins,
                     **kwargs)


@metrics_base.registry.register("ace")
class AdaptiveCalibrationError(GeneralCalibrationError):

  def __init__(self,
               dataset_info: Optional[datasets_base.DatasetInfo] = None,
               num_bins: int = 30,
               **kwargs):
    super().__init__(dataset_info,
                     threshold=0,
                     binning_scheme="adaptive",
                     max_prob=False,
                     class_conditional=True,
                     norm="l1",
                     num_bins=num_bins,
                     **kwargs)


@metrics_base.registry.register("tace")
class ThresholdedAdaptiveCalibrationError(GeneralCalibrationError):

  def __init__(self,
               dataset_info: Optional[datasets_base.DatasetInfo] = None,
               num_bins: int = 30,
               threshold: float = 0.01,
               **kwargs):
    super().__init__(dataset_info,
                     threshold=0,
                     binning_scheme="adaptive",
                     max_prob=False,
                     class_conditional=True,
                     norm="l1",
                     num_bins=num_bins,
                     **kwargs)


@metrics_base.registry.register("msce")
class MonotonicSweepCalibrationError(GeneralCalibrationError):

  def __init__(self,
               dataset_info: Optional[datasets_base.DatasetInfo] = None,
               num_bins: int = 30,
               **kwargs):
    super().__init__(dataset_info,

                     threshold=0,
                     binning_scheme="adaptive",
                     class_conditional=False,
                     max_prob=True,
                     norm="l1",
                     num_bins=None,
                     **kwargs)
