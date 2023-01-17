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

r"""Generate a set of robustness reports on the given model.

The script accepts a path to the python file (including the .py extension),
that should contain:

A function `create` that returns a tuple consisting of
  * the model as a callable: it accepts a dictionary holding batched data and
    returns a tensor of predictions; and
  * the preprocessing as a callable: it is mapped over the dataset before
    batching (use `None` for default preprocessing).

As an example see the file `models/random_imagenet_numpy.py`.

There are two ways how one can specify which metrics on which datasets it should
run:

  - One can explicitly provide them in the flag `--measurement`, e.g., passing
    --measurement=accuracy@imagenet --brier@imagenet_a will compute the accuracy
    of the imagenet dataset and the Brier score over the imagenet_a dataset.
  - By passing a report name in `--report`. The script will figure out which
    combinations of metrics and datasets the report needs and will evaluate all
    of them.

Note that each of these flags can be provided multiple times and the script
evaluates the union of them. All the measurements provided in `--measurement`
will appear under the report with name "custom".

The results are printed to the standard output and can be also saved to a JSON
file using the `--output_json` flag. The json file will hold an array of two
dictionaries (JSON objects):
  * The first one holding a map metric_name -> dataset_name -> result.
  * The second one holding a map report-> report_value_name -> value.

# Note for non-TF models
Please use the flag `--tf_on_cpu`, so that TensorFlow will not allocate any of
the GPU memory.
"""
import json

from absl import app
from absl import flags
import pandas as pd
import robustness_metrics as rm
from robustness_metrics.bin import common as bin_common
from robustness_metrics.bin import compute_report_lib as lib
import tabulate
import tensorflow as tf


FLAGS = flags.FLAGS
flags.DEFINE_multi_string(
    "report", [], "The specifications of the reports to be computed")
flags.DEFINE_multi_string(
    "measurement", [],
    "A @-separated pair specifying which metric to compute on which dataset. "
    "Must be specified if `report` is not specified.")

flags.DEFINE_integer("batch_size", 32, "The batch size to use.")
flags.DEFINE_string("model_path", None,
                    "A path to the python file defining the model.")
flags.DEFINE_string(
    "model_args", "",
    "The arguments to be passed to the create() function of the model. Will "
    "be literal_eval'ed, should take the form a='1',b='2',c=3.")
flags.DEFINE_string("output_json_path", None,
                    "Where to store the json-serialized output")
flags.DEFINE_bool("tf_on_cpu", False,
                  "If set, will hide accelerators from TF.")

flags.mark_flags_as_required(["model_path"])


def _register_custom_report():
  """Registers a report named `custom` from the given --measurement flags."""
  all_measurements = []

  for metric_name, dataset_name in map(lambda spec: spec.split("@"),
                                       FLAGS.measurement):
    all_measurements.append(rm.reports.base.MeasurementSpec(
        dataset_name=dataset_name, metric_name=metric_name))

  @rm.reports.base.registry.register("custom")  # pylint: disable=unused-variable
  class CustomReport(rm.reports.base.UnionReport):

    @property
    def required_measurements(self):
      return all_measurements


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if FLAGS.tf_on_cpu:
    # Hide the GPU from TF.
    tf.config.experimental.set_visible_devices([], "GPU")

  strategy = bin_common.default_distribution_strategy()

  module = bin_common.load_module_from_path(FLAGS.model_path)
  with strategy.scope():
    _, _, kwargs = rm.common.registry.parse_name_and_kwargs(
        f"foo({FLAGS.model_args})")
    model, preprocess_fn = module.create(**kwargs)

  if FLAGS.measurement:
    _register_custom_report()
    FLAGS.report.append("custom")

  metric_results, report_results = lib.compute_reports(
      strategy, FLAGS.report, model, preprocess_fn, FLAGS.batch_size)
  for metric_name, results in metric_results.items():
    print(f"metric: {metric_name}")
    print(tabulate.tabulate(pd.DataFrame.from_dict(results), headers="keys"))

  for report_name, results in report_results.items():
    print(f"report: {report_name}")
    print(tabulate.tabulate(results.items(), headers=["score name", "value"]))

  if FLAGS.output_json_path:
    with tf.io.gfile.GFile(FLAGS.output_json_path, "wb") as json_fp:
      json.dump((metric_results, report_results), json_fp)

if __name__ == "__main__":
  app.run(main)
