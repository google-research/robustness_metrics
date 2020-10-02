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
"""Tools useful for all executable scripts."""
import collections
import importlib
import time
import types as pytypes
from typing import Text, Iterator, Callable, Any, Sequence

from absl import logging
import robustness_metrics as rm
from robustness_metrics.common import types
import tensorflow as tf

PredictionModel = Callable[[types.Features], Any]
PreprocessFn = Callable[[types.Features], types.Features]


def load_module_from_path(model_path: Text) -> pytypes.ModuleType:
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


def parse_reports_names(reports_names: Sequence[Text]):
  """Construct the necessry datasets and metrics for the given reports.

  Args:
    reports_names: The names of the reports.

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
  for report in reports.values():
    for spec in report.required_measurements:
      if spec.dataset_name not in datasets:
        datasets[spec.dataset_name] = rm.datasets.get(spec.dataset_name)
      with tf.device("job:localhost"):
        metric = rm.metrics.get(spec.metric_name,
                                datasets[spec.dataset_name].info)
        metrics[spec.dataset_name][spec.metric_name] = metric
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
  return {key: tensor[i] for key, tensor in tensor_dict.items()}


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
    strategy: tf.distribute.Strategy) -> Iterator[types.ModelPredictions]:
  """Yield the predictions of the model on the given dataset.

  Note that the dataset is expected to yield batches of tensors.

  Args:
    model: A function that takes tensor-valued features and returns a vector of
      predictions.
    dataset: The dataset that the function consumes to produce the predictions.
    strategy: The distribution strategy to use when computing.

  Yields:
    The predictions of the model on the dataset.
  """

  for features in strategy.experimental_distribute_dataset(dataset):
    # TODO(josipd): Figure out how to pass only tpu-allowed types.
    time_start = time.time()
    predictions = materialize(
        strategy, strategy.run(model, args=({
            "image": features["image"]
        },)))
    time_end = time.time()
    time_delta_per_example = (time_end - time_start) / predictions.shape[0]
    try:
      element_ids = materialize(strategy, features["element_id"])
    except KeyError:
      element_ids = [None] * predictions.shape[0]
    metadatas = materialize(strategy, features["metadata"])
    for i in range(predictions.shape[0]):
      yield types.ModelPredictions(
          element_id=element_ids[i],
          metadata=_slice_dictionary(metadatas, i),
          predictions=[predictions[i]],
          time_in_s=time_delta_per_example)
