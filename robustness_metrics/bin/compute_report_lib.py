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

"""The reusable components."""
import collections
from typing import Dict, Sequence, Optional
from robustness_metrics.bin import common as bin_common
import tensorflow as tf


def compute_reports(strategy: tf.distribute.Strategy,
                    reports_names: Sequence[str],
                    model: bin_common.PredictionModel,
                    preprocess_fn: Optional[bin_common.PreprocessFn],
                    batch_size: int,
                    measurements: Optional[Dict[str, Sequence[str]]] = None):
  """Compute the report and return the results.

  Args:
    strategy: The distribution strategy to use when computing.
    reports_names: The names of the reports to run.
    model: The model that will be evaluated.
    preprocess_fn: Used to preprocess the dataset before batching.
    batch_size: The batch size to use when computing the predictions.
    measurements: An optional set of metrics to compute. The dictionary is
      expected to map dataset names to the list of metrics to compute on them.
  Returns:
    A dictionary metric name -> dataset_name -> result.
    A dictionary report name -> result dictionary.
  """
  reports, metrics, datasets = bin_common.parse_reports_names(reports_names,
                                                              measurements)

  for dataset_name, dataset in datasets.items():
    tf_dataset = dataset.load(preprocess_fn=preprocess_fn)
    for predictions, metadata in bin_common.compute_predictions(model,
                                                                tf_dataset,
                                                                strategy,
                                                                batch_size):
      with tf.device("job:localhost"):
        for metric in metrics[dataset_name].values():
          metric.add_predictions(predictions, metadata)

  metric_results = collections.defaultdict(dict)
  for dataset_name, metrics_dict in metrics.items():
    for metric_name, metric in metrics_dict.items():
      metric_results[metric_name][dataset_name] = metric.result()

  report_results = {}
  for report_name, report in reports.items():
    for spec in report.required_measurements:
      report.add_measurement(
          spec.dataset_name, spec.metric_name,
          metric_results[spec.metric_name][spec.dataset_name])
    report_results[report_name] = report.result()

  return metric_results, report_results
