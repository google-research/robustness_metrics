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

"""Tools useful for all executable scripts."""
import collections
import importlib
import math
import time
import types as pytypes
from typing import Any, Callable, Iterator, Tuple, Sequence, Optional, Dict

from absl import logging
import clu.deterministic_data as clu_dd
import jax
import numpy as np
import robustness_metrics as rm
from robustness_metrics.common import types
import tensorflow as tf

PredictionModel = Callable[[types.Features], Any]
PreprocessFn = Callable[[types.Features], types.Features]


def load_module_from_path(model_path: str) -> pytypes.ModuleType:
  """Load the Python module at the given path.

  Args:
    model_path: The full path to file holding the code for the module.

  Returns:
    The module loaded from the given path.
  """
  module_spec = importlib.util.spec_from_file_location("model", model_path)
  module = importlib.util.module_from_spec(module_spec)
  module_spec.loader.exec_module(module)  # pytype: disable=attribute-error
  return module


def parse_reports_names(
    reports_names: Sequence[str],
    measurements: Optional[Dict[str, Sequence[str]]] = None):
  """Construct the necessry datasets and metrics for the given reports.

  Args:
    reports_names: The names of the reports.
    measurements: An optional set of metrics to compute. The dictionary is
      expected to map dataset names to the list of metrics to compute on them.

  Returns:
    A mapping from the report name to the corresponding Report objects.
    A mapping dataset name -> metric_name -> Metric object.
    A mapping from the name of the dataset to the tf.Dataset object.
  """
  # Note that we place all tensors that the metrics and reports initialize on
  # the local device. This is so that we do not have to update remote tensors
  # when we add new measurements.
  with tf.device("job:localhost"):
    reports = {
        report_name: rm.reports.get(report_name)
        for report_name in reports_names
    }
  # We store the following dict: dataset_name -> metric_name -> metric
  metrics = collections.defaultdict(dict)
  datasets = {}

  def add_dataset_and_metric(dataset_name, metric_name):
    if dataset_name not in datasets:
      datasets[dataset_name] = rm.datasets.get(dataset_name)
    with tf.device("job:localhost"):
      metric = rm.metrics.get(metric_name,
                              datasets[dataset_name].info)
      metrics[dataset_name][metric_name] = metric

  for report in reports.values():
    for spec in report.required_measurements:
      add_dataset_and_metric(spec.dataset_name, spec.metric_name)

  if measurements:
    for dataset_name, metric_names in measurements.items():
      for metric_name in metric_names:
        add_dataset_and_metric(dataset_name, metric_name)

  return reports, metrics, datasets


def default_distribution_strategy():
  for device_type in ["GPU", "CPU"]:
    devices = tf.config.experimental.list_logical_devices(
        device_type=device_type)
    if len(devices) > 1:
      logging.info("Using devices: %r", devices)
      return tf.distribute.MirroredStrategy(devices=devices)
    elif len(devices) == 1:
      logging.info("Using device: %r", devices[0])
      return tf.distribute.OneDeviceStrategy(device=devices[0])


def _slice_dictionary(tensor_dict, i: int):
  return jax.tree_map(lambda x: x[i], tensor_dict)


def materialize(strategy: tf.distribute.Strategy, value_or_nested_dict):
  """Materializes locally (possibly nested dict with) PerReplica values.

  Args:
    strategy: The strategy that will be used to evaluate.
    value_or_nested_dict: Either a single `PerReplica` object, or a nested dict
      with `PerReplica` values at the deepest level.

  Returns:
    Same type and format as the input, with PerReplica values replaced with
    corresponding `tf.Tensor`s.
  """
  if isinstance(value_or_nested_dict, dict):
    nested_dict = value_or_nested_dict
    return {
        key: materialize(strategy, value) for key, value in nested_dict.items()
    }
  else:
    return tf.concat(
        strategy.experimental_local_results(value_or_nested_dict),
        axis=0).numpy()


def compute_predictions(
    model: PredictionModel, dataset: tf.data.Dataset,
    strategy: tf.distribute.Strategy, batch_size: int
) -> Iterator[Tuple[types.ModelPredictions, types.Features]]:
  """Yield the predictions of the model on the given dataset.

  Args:
    model: A function that takes tensor-valued features and returns a vector of
      predictions.
    dataset: The dataset that the function consumes to produce the predictions.
    strategy: The distribution strategy to use when computing.
    batch_size: The batch size that should be used.

  Yields:
    Pairs of model predictions and the corresponding metadata.
  """
  with strategy.scope():
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA)
    dataset = dataset.with_options(options)

  for features in strategy.experimental_distribute_dataset(dataset):
    time_start = time.time()
    if isinstance(strategy, tf.distribute.experimental.TPUStrategy):
      # TODO(josipd): Figure this out better. We can't easily filter,
      #               as they are PerReplica values, not tensors.
      features_model = {"image": features["image"]}
    else:
      features_model = features
    predictions = materialize(strategy,
                              strategy.run(model, args=(features_model,)))
    time_end = time.time()
    time_delta_per_example = (time_end - time_start) / predictions.shape[0]
    metadatas = materialize(strategy, features["metadata"])
    for i in range(predictions.shape[0]):
      model_predictions = types.ModelPredictions(
          predictions=[predictions[i]],
          time_in_s=time_delta_per_example)
      metadata_i = _slice_dictionary(metadatas, i)
      yield model_predictions, metadata_i


def compute_predictions_jax(
    model: PredictionModel, dataset: tf.data.Dataset, batch_size: int
)-> Iterator[Tuple[types.ModelPredictions, types.Features]]:
  """Yield the predictions of the given JAX model on the given dataset.

  Note that this also works in multi-host configurations. You have to make
  sure that this function gets called on all hosts. The results will be yielded
  only to the host with a jax.host_id() equal to 0.

  Args:
    model: A function that takes tensor-valued features and returns a vector of
      predictions.
    dataset: The dataset that the function consumes to produce the predictions.
    batch_size: The batch size that should be used.

  Yields:
    The predictions of the model on the dataset.
  """

  def _gather(inputs):
    return jax.lax.all_gather(inputs, "i")
  gather = jax.pmap(_gather, axis_name="i")

  def reshape_for_pmap(array):
    return jax.tree_map(
        lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]),
        array)

  def infer(features):
    probabilities = model(features)
    return_vals = (probabilities, features["mask"])
    return_vals_reshaped = reshape_for_pmap(return_vals)
    return jax.tree_map(lambda x: x[0], gather(return_vals_reshaped))

  def gather_metadata(features):
    return_vals = (features["metadata"],)
    return_vals_reshaped = reshape_for_pmap(return_vals)
    return jax.tree_map(lambda x: x[0], gather(return_vals_reshaped))

  if dataset.cardinality() < 0:
    raise ValueError(
        "The cardinality must be known when running JAX multi-host models.")
  total_batches = math.ceil(dataset.cardinality() / batch_size)
  lcm = lambda x, y: (x * y) // math.gcd(x, y)
  # We want each shard (host) to get an equal number of batches.
  total_batches_padded = lcm(jax.host_count(), total_batches)
  logging.info("Total batches %d, rounded up to %d",
               total_batches, total_batches_padded)

  def pad_strings(array):
    if array.dtype != tf.string:
      return array
    array_bytes = tf.strings.unicode_decode(array, "UTF-8")
    # The return type is either Tensor or RaggedTensor.
    try:
      # When a RaggedTensor, which we need to convert it.
      # to_tensor() adds a leading dimension of size 1, which we drop.
      array_bytes = array_bytes.to_tensor()[0]
    except AttributeError:
      pass
    array_size = tf.size(array_bytes)
    with tf.control_dependencies([
        tf.compat.v1.assert_less_equal(array_size, 1024)]):
      packed = tf.pad(array_bytes, [[0, 1024 - array_size]])
    return {"__packed": tf.ensure_shape(packed, [1024])}

  def unpad_strings(array):
    if isinstance(array, dict):
      with_trailing_zeros = bytes(tf.strings.unicode_encode(
          np.asarray(array["__packed"]).reshape(-1), "UTF-8").numpy())
      return with_trailing_zeros.rstrip(b"\x00")
    else:
      return np.asarray(array)

  def pad_strings_in_metadata(features):
    """Only padding of the strings subject to a gather operation."""
    features["metadata"] = tf.nest.map_structure(pad_strings,
                                                 features["metadata"])
    return features

  dataset = clu_dd.pad_dataset(
      dataset.map(pad_strings_in_metadata),
      batch_dims=[batch_size],
      pad_up_to_batches=total_batches_padded,
      cardinality=None,  # It will be inferred from the datset.
  ).batch(batch_size)

  # The shard for the current host.
  dataset_shard = dataset.shard(jax.host_count(), jax.host_id())
  logging.info("Batches per host: %d", dataset_shard.cardinality())
  for features in dataset_shard.as_numpy_iterator():
    time_start = time.time()
    # There is a bug in XLA, the following fails for int8s.
    features["mask"] = features["mask"].astype(np.int32)

    flatten = lambda array: array.reshape((-1,) + array.shape[2:])
    predictions, masks = jax.tree_map(flatten, infer(features))
    with jax.experimental.enable_x64():  # pytype: disable=module-attr
      metadatas = jax.tree_map(flatten, gather_metadata(features))[0]

    time_end = time.time()
    time_delta_per_example = (time_end - time_start) / predictions.shape[0]
    predictions = np.asarray(predictions)  # Materialize.
    if jax.host_id() == 0:
      for i in range(predictions.shape[0]):
        if masks[i]:
          predictions_i = types.ModelPredictions(
              predictions=[predictions[i]], time_in_s=time_delta_per_example)
          with jax.experimental.enable_x64():  # pytype: disable=module-attr
            metadata_i = _slice_dictionary(metadatas, i)
            is_leaf_fn = lambda x: isinstance(x, dict) and "__packed" in x
            metadata_i_unpadded = jax.tree_map(
                unpad_strings, metadata_i, is_leaf=is_leaf_fn)
          _check_element_id_is_an_int64(metadata_i_unpadded)
          yield predictions_i, metadata_i_unpadded


def _check_element_id_is_an_int64(metadata):
  """We check that the hash in metadata['element_id'] is an int64."""
  assert_msg = ("metadata['element_id'] must be an int64, but "
                f"received {metadata['element_id'].dtype!r} instead.")
  assert metadata["element_id"].dtype == jax.numpy.int64, assert_msg
