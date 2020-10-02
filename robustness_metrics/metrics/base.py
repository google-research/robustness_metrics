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
"""The base Metric class and an accuracy metric."""

import abc
import collections
from typing import Dict, Text

import numpy as np
from robustness_metrics.common import registry
from robustness_metrics.common import types
import tensorflow as tf


class Metric(metaclass=abc.ABCMeta):
  """The abstract class representing a metric.

  Each metric receives a set of predictions via calls to the `add_predictions`
  method, and computes the results using the `result` method.

  The function `result` might be called several times, and the results should
  be always computed using the data received over the complete life-time of
  the object.
  """

  def __init__(self, dataset_info=None):
    """Initializes the metric.

    Args:
      dataset_info: A datasets.DatasetInfo object.
    """

  @abc.abstractmethod
  def add_predictions(self, model_predictions: types.ModelPredictions) -> None:
    """Adds a new prediction that will be used when computing the metric.

    Args:
      model_predictions: The predictions that the model made on an element
        from the dataset.
    """

  @abc.abstractmethod
  def result(self) -> Dict[Text, float]:
    """Computes the results from all the predictions it has seen so far.

    Returns:
      A dictionary mapping the name of each computed metric to its value.
    """


registry = registry.Registry(Metric)
get = registry.get


class KerasMetric(Metric):
  """Wraps a KerasMetric to accept ModelPredictions.

  The arguments specify how the predictions are labels are processed before
  being passed to the `update_state` method of the wrapped metric.
  """

  def __init__(self,
               dataset_info,
               keras_metric: tf.keras.metrics.Metric,
               key_name: str,
               take_argmax: bool = False,
               one_hot: bool = False,
               average_predictions: bool = True):
    """Initializes the object.

    Args:
      dataset_info: The DatasetInfo object associated with the dataset.
      keras_metric: The metric being wrapped.
      key_name: The key under which the result from the metric will be reported.
        In code, the reult will be `{key: wrapped_metric.result()}`.
      take_argmax: If set, the argmax of the predictions will be sent to the
        metric rather than the predictions themselves.
      one_hot: If set, the label will be one-hot encoded.
      average_predictions: If set, when multiple predictions are present for
        a dataset element, they will be averaged before processing.
    """
    super().__init__(dataset_info)
    self._metric = keras_metric
    self._key_name = key_name
    self._take_argmax = take_argmax
    self._one_hot = one_hot
    self._average_predictions = average_predictions
    self._num_classes = dataset_info.num_classes if dataset_info else None
    self._ids_seen = set()

  def _add_prediction(self, label, predictions):
    """Feeds the given label and prediction to the underlying Keras metric."""
    if self._average_predictions:
      predictions = tf.reduce_mean(predictions, axis=0, keepdims=True)

    if self._one_hot:
      label = tf.one_hot(label, self._num_classes)
    if self._take_argmax:
      self._metric.update_state(label, tf.argmax(predictions, axis=-1))
    else:
      self._metric.update_state(label, predictions)

  def add_predictions(self, model_predictions: types.ModelPredictions) -> None:
    if model_predictions.element_id is not None:
      element_id = int(model_predictions.element_id)
      if element_id in self._ids_seen:
        raise ValueError(
            "KerasMetric does not support reporting the same id multiple "
            f"times, but you added element id {element_id!r} twice.")
      else:
        self._ids_seen.add(element_id)
    stacked_predictions = np.stack(model_predictions.predictions)
    if "label" not in model_predictions.metadata:
      raise ValueError(
          "KerasMetric expects a `label` in the `metadata` field."
          f"Available fields are: {model_predictions.metadata.keys()!r}")
    self._add_prediction(model_predictions.metadata["label"],
                         stacked_predictions)

  def result(self) -> Dict[Text, float]:
    return {self._key_name: float(self._metric.result())}


@registry.register("accuracy")
class Accuracy(KerasMetric):
  """Computes the average prediction.

  This metric computes a dictionary with a single key "accuracy" holding as
  a value the top-1 accuracy. The true zero-indexed label must be provided in
  the metadata field `"label"`.
  """

  def __init__(self, dataset_info=None):
    metric = tf.keras.metrics.Accuracy()
    super().__init__(dataset_info, metric, "accuracy",
                     take_argmax=True, one_hot=False)


class AggregatedAccuracy(Metric):
  """Computes the average of per-group aggregated scores.

  This metric assumes that the elements can be grouped and accepts an aggregator
  function to compute the score per each group. It returns the average across
  all groups. One example usage is for computing the stability metrics.
  """

  def __init__(self,
               group_element_id_field,
               aggregator_fn,
               dataset_info=None):
    """Initializes the AggregatedAccuracy metric.

    Args:
      group_element_id_field: The name of the field in the metadata holding a
        '/'-separated utf-8 encoded string containing an identifier for the
        group and the identifier for the element. For example, one could have
        `video_id/element_id` and the metric would group according to the
        `video_id` substring.
      aggregator_fn: This function is applied to prediction results (correct or
        not) for each instance of the group.
      dataset_info: DatasetInfo object containing useful information for any
        subclass.
    """
    super().__init__(dataset_info)
    self._groups = collections.defaultdict(list)
    self._group_element_id_field = group_element_id_field
    self._aggregator_fn = aggregator_fn
    self._dataset_info = dataset_info
    self._classes_to_ignore = None
    if self._dataset_info:
      if self._dataset_info.appearing_classes:
        num_classes = self._dataset_info.num_classes
        all_classes = set(range(num_classes))
        appearing_classes = set(self._dataset_info.appearing_classes)
        self._classes_to_ignore = all_classes.difference(appearing_classes)

  def get_groups(self):
    """Returns a dictionary mapping each group to tuples of (element, score)."""
    return self._groups

  def add_predictions(self, model_predictions: types.ModelPredictions) -> None:
    # We distinguish two cases, `multi-hot-labeled` and single label cases.
    # If both `multi_hot_label` and `label` are are available, `label` is used.
    if ("label" in model_predictions.metadata and
        "labels_multi_hot" in model_predictions.metadata):
      raise ValueError("The dataset element can't have `label` and "
                       "`labels_multi_hot` simultaneously.")
    is_multi_label = "label" not in model_predictions.metadata
    if is_multi_label and "labels_multi_hot" not in model_predictions.metadata:
      raise ValueError("Required field is missing: `labels_multi_hot`.")
    # We assume that the group is a substring until the first `/` and
    # everything that follows is the identifying the element.
    field_str = model_predictions.metadata[self._group_element_id_field]
    if isinstance(field_str, bytes):
      field_str = field_str.decode("utf-8")
    group_id, element_id = field_str.split("/")

    for prediction in model_predictions.predictions:
      if self._classes_to_ignore:
        for x in self._classes_to_ignore:
          prediction[x] = -np.infty

      if is_multi_label:
        correct = np.where(model_predictions.metadata["labels_multi_hot"])[0]
        predicted = np.argmax(prediction)
        self._groups[group_id].append((element_id, int(predicted in correct)))
      else:
        correct = model_predictions.metadata["label"]
        predicted = np.argmax(prediction)
        self._groups[group_id].append((element_id, int(predicted == correct)))

  def result(self) -> Dict[Text, float]:
    group_scores = []
    for _, element_scores in self._groups.items():
      # The default aggregator ignores the element IDs.
      scores = [score[1] for score in element_scores]
      group_scores.append(self._aggregator_fn(scores))
    return {"aggregated_accuracy": np.mean(group_scores)}


@registry.register("precision")
class Precision(KerasMetric):
  """Computes the precision.

  If `top_k` is set, we'll calculate precision as how often on average a class
  among the top-k classes with the highest predicted values is correct (can be
  found in the label for that entry). For example, if classes `[1, 3, 5]` are
  correct and our top 5 predictions are `[1, 2, 3, 4, 5]` the precision is `.6`.
  """

  def __init__(self, top_k=None, dataset_info=None):
    metric = tf.keras.metrics.Precision(top_k=top_k)

    super().__init__(
        dataset_info=dataset_info,
        keras_metric=metric,
        key_name=f"precision@{top_k}",
        one_hot=False,  # We are making sure we always pass the one-hot labels.
        take_argmax=False)

  @tf.function
  def parse_label(self, class_id):
    class_id = tf.cast([class_id], tf.int64)
    return tf.one_hot(class_id, self._num_classes)

  def add_predictions(self, model_predictions: types.ModelPredictions) -> None:
    # Wrapping label into size-1 batch.
    if "label" in model_predictions.metadata:
      label = self.parse_label(model_predictions.metadata["label"])
    else:
      label = [model_predictions.metadata["labels_multi_hot"]]
    super().add_predictions(types.ModelPredictions(
        model_predictions.element_id,
        {"label": label},
        model_predictions.predictions))


@registry.register("accuracy_top_k")
class TopKAccuracy(Metric):
  """Top-k Accuracy.

  A set of predictions are considered to be correct, if any of the coordinates
  holding the k largest values contains the correct label. Note that in the case
  of multiple labels a prediction is considred to be correct if *any* of the
  labels is predicted.

  At the moment there is only support for multi-one-hot encoded labels in the
  "labels_multi_hot" field of the metadata.
  """

  def __init__(self, top_k: int, dataset_info):
    """Initializes the object.

    Args:
      top_k: How many positions to consider, i.e., the k in top-k.
      dataset_info: The DatasetInfo object associated with the dataset.
    """
    super().__init__(dataset_info)
    self._num_classes = dataset_info.num_classes
    self._mean = tf.keras.metrics.Mean()
    self._top_k = top_k

  @tf.function
  def _add_prediction(self, labels, predictions):
    _, top_k_indices = tf.math.top_k(predictions, self._top_k)
    top_k_labels = tf.gather(labels, top_k_indices, batch_dims=0)
    top_k_correct = tf.reduce_any(top_k_labels > 0, axis=-1)
    self._mean.update_state(tf.reduce_mean(tf.cast(top_k_correct, tf.float32)))

  def add_predictions(self, model_predictions):
    stacked_predictions = np.stack(model_predictions.predictions)
    self._add_prediction(
        model_predictions.metadata["labels_multi_hot"],
        tf.reduce_mean(stacked_predictions, axis=0))

  def result(self):
    return {f"accuracy@{self._top_k}": float(self._mean.result().numpy())}
