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
"""The base Metric class and an accuracy metric."""

import abc
import collections
import operator
from typing import Any, Dict, List, Optional, Text, Tuple

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
  def add_predictions(self,
                      model_predictions: types.Array,
                      metadata: types.Features) -> None:
    """Adds predictions for a single example.

    Multiple predictions for a single example can be added depending on the
    metric. In order to add predictions for multiple examples, use
    `add_batch()`.

    Args:
      model_predictions: The predictions that the model made on an element from
        the dataset. Array with shape [...]. Some metric subclasses may require
        a shape [num_predictions, ...] where they evaluate over multiple
        predictions for a given example.
      metadata: The metadata for the example.
    """

  def add_batch(self,
                model_predictions: types.Array,
                **metadata: Optional[types.Features]) -> None:
    """Adds predictions for a batch of examples.

    Args:
      model_predictions: The batch of predictions. Array with shape [batch_size,
        ...] where [...] is the shape of a single prediction. Some metric
        subclasses may require a shape [batch_size, num_predictions, ...] where
        they evaluate over multiple predictions per example.
      **metadata: The batch metadata, possibly including `label` which is the
        batch of labels, one for each example in the batch. Each metadata kwarg
        must be batched such as `label` with shape [batch_size].
    """
    def _recursive_map(fn, dict_or_val):
      if isinstance(dict_or_val, dict):
        return {k: _recursive_map(fn, v) for k, v in dict_or_val.items()}
      else:
        return fn(dict_or_val)

    for i, predictions_i in enumerate(np.array(model_predictions)):
      metadata_i = _recursive_map(operator.itemgetter(i), metadata)
      self.add_predictions(predictions_i, metadata_i)

  @abc.abstractmethod
  def result(self) -> Dict[Text, float]:
    """Computes the results from all the predictions it has seen so far.

    Returns:
      A dictionary mapping the name of each computed metric to its value.
    """


registry = registry.Registry(Metric)
get = registry.get


def _map_labelset(predictions: types.Array,
                  label: types.Array,
                  appearing_classes) -> Tuple[np.ndarray, List[Any]]:
  np_predictions = np.array(predictions)
  assert len(np_predictions.shape) >= 1
  if appearing_classes:
    predictions = np_predictions[..., appearing_classes]
    predictions /= np.sum(predictions, axis=-1, keepdims=True)
    label = appearing_classes.index(label)
  return predictions, label


class FullBatchMetric(Metric):
  """Base class for metrics that operate on the full dataset (not streaming)."""

  def __init__(self, dataset_info=None, use_dataset_labelset=False):
    self._ids_seen = set()
    self._predictions = []
    self._labels = []
    self._use_dataset_labelset = use_dataset_labelset
    self._appearing_classes = (dataset_info.appearing_classes if dataset_info
                               else None)
    super().__init__(dataset_info)

  def add_predictions(self,
                      model_predictions: types.Array,
                      metadata: types.Features) -> None:
    try:
      element_id = int(metadata["element_id"])
      if element_id in self._ids_seen:
        raise ValueError(f"You added element id {element_id!r} twice.")
      else:
        self._ids_seen.add(element_id)
    except KeyError:
      pass

    try:
      label = metadata["label"]
    except KeyError:
      raise ValueError("No labels in the metadata, provided fields: "
                       f"{metadata.keys()!r}")
    if self._use_dataset_labelset:
      model_predictions, label = _map_labelset(
          model_predictions, label, self._appearing_classes)
    self._predictions.append(model_predictions)
    self._labels.append(label)


class KerasMetric(Metric):
  """Wraps a Keras metric to be a `rm.Metric`.

  The arguments specify how the predictions and labels are processed before
  being passed to the `update_state` method of the wrapped metric.
  """

  def __init__(self,
               dataset_info,
               keras_metric: tf.keras.metrics.Metric,
               key_name: str,
               take_argmax: bool = False,
               one_hot: bool = False,
               use_dataset_labelset: bool = False):
    """Initializes the object.

    Args:
      dataset_info: The DatasetInfo object associated with the dataset.
      keras_metric: The metric being wrapped.
      key_name: The key under which the result from the metric will be reported.
        In code, the reult will be `{key: wrapped_metric.result()}`.
      take_argmax: If set, the argmax of the predictions will be sent to the
        metric rather than the predictions themselves.
      one_hot: If set, the label will be one-hot encoded.
      use_dataset_labelset: If set, and the given dataset has only a subset of
        the clases the model produces, the classes that are not in the dataset
        will be removed and the others scaled to sum up to one.
    """
    super().__init__(dataset_info)
    self._metric = keras_metric
    self._key_name = key_name
    self._take_argmax = take_argmax
    self._one_hot = one_hot
    if dataset_info:
      self._appearing_classes = dataset_info.appearing_classes
      if self._appearing_classes and use_dataset_labelset:
        self._num_classes = len(self._appearing_classes)
      else:
        self._num_classes = dataset_info.num_classes
    else:
      self._num_classes = None
      self._appearing_classes = None
    self._ids_seen = set()
    self._use_dataset_labelset = use_dataset_labelset

  @tf.function
  def _add_prediction(self,
                      predictions: types.Array,
                      label: types.Array) -> None:
    """Feeds the given label and prediction to the underlying Keras metric."""
    if self._one_hot:
      label = tf.one_hot(label, self._num_classes)
    if self._take_argmax:
      self._metric.update_state(label, tf.argmax(predictions, axis=-1))
    else:
      self._metric.update_state(label, predictions)

  def add_predictions(self,
                      model_predictions: types.Array,
                      metadata: types.Features) -> None:
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

    if "label" not in metadata:
      raise ValueError(
          "KerasMetric expects a `label` in the metadata."
          f"Available fields are: {metadata.keys()!r}")

    # Add a size-1 batch axis to be compatible with Keras metrics.
    predictions = np.expand_dims(model_predictions, axis=0)
    if self._use_dataset_labelset:
      predictions, label = _map_labelset(
          predictions, metadata["label"], self._appearing_classes)
      self._add_prediction(predictions, label)
    else:
      self._add_prediction(predictions, metadata["label"])

  def add_batch(self,
                model_predictions: types.Array,
                **metadata: Optional[types.Features]) -> None:
    # Note that even though the labels are really over a batch of predictions,
    # we use the kwarg "label" to be consistent with the other functions that
    # use the singular name.
    label = metadata["label"]
    if self._use_dataset_labelset:
      model_predictions = tf.gather(
          model_predictions, self._appearing_classes, axis=-1)
      model_predictions /= tf.math.reduce_sum(
          model_predictions, axis=-1, keepdims=True)
      label = tf.convert_to_tensor([
          self._appearing_classes.index(x) for x in label])
    self._add_prediction(model_predictions, label)

  def reset_states(self):
    return self._metric.reset_states()

  def result(self) -> Dict[Text, float]:
    return {self._key_name: float(self._metric.result())}


@registry.register("accuracy")
class Accuracy(KerasMetric):
  """Computes the average prediction.

  This metric computes a dictionary with a single key "accuracy" holding as
  a value the top-1 accuracy. The true zero-indexed label must be provided in
  the metadata field `"label"`.
  """

  def __init__(self, dataset_info=None, use_dataset_labelset=False):
    metric = tf.keras.metrics.Accuracy()
    super().__init__(dataset_info, metric, "accuracy",
                     take_argmax=True, one_hot=False,
                     use_dataset_labelset=use_dataset_labelset)


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

  def add_predictions(self,
                      model_predictions: types.Array,
                      metadata: types.Features) -> None:
    # We distinguish two cases, `multi-hot-labeled` and single label cases.
    # If both `multi_hot_label` and `label` are are available, `label` is used.
    if "label" in metadata and "labels_multi_hot" in metadata:
      raise ValueError("The dataset element can't have `label` and "
                       "`labels_multi_hot` simultaneously.")
    is_multi_label = "label" not in metadata
    if is_multi_label and "labels_multi_hot" not in metadata:
      raise ValueError("Required field is missing: `labels_multi_hot`.")
    # We assume that the group is a substring until the first `/` and
    # everything that follows is the identifying the element.
    field_str = metadata[self._group_element_id_field]
    if isinstance(field_str, bytes):
      field_str = field_str.decode("utf-8")
    group_id, element_id = field_str.split("/")

    if self._classes_to_ignore:
      for x in self._classes_to_ignore:
        model_predictions[x] = -np.infty

    if is_multi_label:
      correct = np.where(metadata["labels_multi_hot"])[0]
      predicted = np.argmax(model_predictions)
      self._groups[group_id].append((element_id, int(predicted in correct)))
    else:
      correct = metadata["label"]
      predicted = np.argmax(model_predictions)
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

  def add_predictions(self,
                      model_predictions: types.Array,
                      metadata: types.Features) -> None:
    # TODO(trandustin): Should this be moved into KerasMetric overall? Does
    # Accuracy work as-is?
    # Wrap label into size-1 batch to be plugged into Keras metric.
    try:
      label = self.parse_label(metadata["label"])
    except KeyError:
      label = [metadata["labels_multi_hot"]]
    metadata = dict(metadata)
    metadata["label"] = label
    super().add_predictions(model_predictions, metadata=metadata)


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

  def __init__(self, top_k: int, dataset_info, use_dataset_labelset=False):
    """Initializes the object.

    Args:
      top_k: How many positions to consider, i.e., the k in top-k.
      dataset_info: The DatasetInfo object associated with the dataset.
      use_dataset_labelset: If set, and the given dataset has only a subset of
        the classes the model produces, the classes that are not in the dataset
        will be removed.
    """
    super().__init__(dataset_info)
    self._num_classes = dataset_info.num_classes
    self._mean = tf.keras.metrics.Mean()
    self._top_k = top_k
    self._use_dataset_labelset = use_dataset_labelset
    self._appearing_classes = dataset_info.appearing_classes

  @tf.function
  def _add_prediction(self,
                      predictions: types.Array,
                      labels: types.Array) -> None:
    """Feeds the given label and prediction to the underlying Keras metric."""
    if self._use_dataset_labelset:
      predictions = tf.gather(predictions, self._appearing_classes, axis=-1)
      labels = tf.gather(labels, self._appearing_classes)
    _, top_k_indices = tf.math.top_k(predictions, self._top_k)
    top_k_labels = tf.gather(labels, top_k_indices, batch_dims=0)
    top_k_correct = tf.reduce_any(top_k_labels > 0, axis=-1)
    self._mean.update_state(tf.reduce_mean(tf.cast(top_k_correct, tf.float32)))

  def add_predictions(self,
                      model_predictions: types.Array,
                      metadata: types.Features) -> None:
    self._add_prediction(model_predictions, metadata["labels_multi_hot"])

  def add_batch(self,
                model_predictions: types.Array,
                **metadata: Optional[types.Features]) -> None:
    self._add_prediction(model_predictions, metadata["label"])

  def result(self):
    return {f"accuracy@{self._top_k}": float(self._mean.result().numpy())}
