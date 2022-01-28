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

# Lint as: python3
r"""Save the model outputs to a file on disk.

The predictions can be loaded into a model using `models/serialized.py`.

The flags and usage is similar to that of `bin/compute_report.py`, the main
difference being that this script accepts a Python module specification instead
of a Python file.

The datasets on which the predictions will be made are read from `--report`.

# Parallelization

You can parallelize the worload by launching this script multiple times and
setting `--num_workers` and `--worker_index`.

# Note for non-TF models
Please use the flag `--tf_on_cpu`, so that TensorFlow will not allocate any of
the GPU memory.

# Example usage
```
python3 bin/serialize.py \
   --model_module=robustness_metrics.models.random_imagenet_numpy \
   --report=imagenet_variants \
   --output_dir=gs://your_bucket/model_outputs
```
"""
import gc
import importlib
import io
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import robustness_metrics as rm
from robustness_metrics.bin import common as bin_common
import tensorflow as tf


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "report", None,
    "The report whose datasets will be used to compute the predictions.")

flags.DEFINE_integer("batch_size", 32, "The batch size to use.")
flags.DEFINE_string("model_module", None,
                    "A python module file defining the model.")
flags.DEFINE_string(
    "model_args", "",
    "The arguments to be passed to the create() function of the model. Will "
    "be literal_eval'ed, should take the form a='1',b='2',c=3.")
flags.DEFINE_string("output_dir", None,
                    "Where to store the dense output. Can be anything that "
                    "can be open with tf.io, including Google cloud buckets.")
flags.DEFINE_bool("tf_on_cpu", True,
                  "If set, will hide accelerators from TF.")
flags.DEFINE_integer(
    "num_workers", 1,
    "How many instances of the script are launched to parallelize compute "
    "across datasets.")
flags.DEFINE_integer("worker_index", 0, "The zero-indexed id of this worker.")

flags.mark_flags_as_required(["model_module"])


def slice_per_worker(dictionary):
  """Equally partitions input over workers. Remainder is given to last worker.

  Args:
    dictionary: dict to split over.

  Returns:
    Assigned dictionary split for the worker index.
  """
  keys_sorted = sorted(dictionary)
  block_size = len(keys_sorted) / FLAGS.num_workers
  start_index = int(FLAGS.worker_index * block_size)
  end_index = int((FLAGS.worker_index + 1) * block_size)
  if FLAGS.worker_index + 1 == FLAGS.num_workers:
    end_index = len(keys_sorted)
  return {key: dictionary[key] for key in keys_sorted[start_index:end_index]}


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if FLAGS.tf_on_cpu:
    # Hide the GPU from TF.
    tf.config.experimental.set_visible_devices([], "GPU")

  strategy = bin_common.default_distribution_strategy()

  module = importlib.import_module(FLAGS.model_module)
  with strategy.scope():
    _, _, kwargs = rm.common.registry.parse_name_and_kwargs(
        f"foo({FLAGS.model_args})")
    model, preprocess_fn = module.create(**kwargs)

  reports, metrics, datasets = bin_common.parse_reports_names([FLAGS.report])
  del reports
  del metrics
  datasets_worker = slice_per_worker(datasets)
  logging.info("All datsets %s", list(datasets))
  logging.info("Worker datsets %s", list(datasets_worker))

  tf.io.gfile.mkdir(FLAGS.output_dir)

  for dataset_name, dataset in datasets_worker.items():
    logging.info("Processing dataset %s", dataset_name)
    # Restore some memory if possible.
    gc.collect()

    element_ids = []
    labels = []
    pred_probs = []

    tf_dataset = dataset.load(preprocess_fn=preprocess_fn)

    cardinality = tf.data.experimental.cardinality(tf_dataset).numpy()

    processed = 0
    for predictions, metadata in bin_common.compute_predictions(
        model, tf_dataset, strategy, batch_size=FLAGS.batch_size):
      element_ids.append(int(metadata["element_id"]))
      try:
        labels.append(int(metadata["label"]))
      except KeyError:  # Not all datasets have labels.
        pass
      pred_probs.append(np.mean(predictions.predictions, axis=0))
      processed += 1
      if processed % 5000 == 0:
        if cardinality < 0:
          logging.info("Processed %d examples", processed)
        else:
          logging.info("Processed %d examples, %f%%",
                       processed, 100 * float(processed) / cardinality)

    element_ids_np = np.stack(element_ids)
    if labels:
      labels_np = np.stack(labels)
    else:
      labels_np = []
    pred_probs_np = np.stack(pred_probs)
    output_path = os.path.join(FLAGS.output_dir, dataset_name + ".npz")
    logging.info("Saving to %s", output_path)
    with tf.io.gfile.GFile(output_path, "wb") as fp:
      with io.BytesIO() as bytes_io:
        # TODO(josipd): Add human-readable names.
        np.savez(bytes_io, element_ids_np, labels_np, pred_probs_np)
        bytes_io.seek(0)
        fp.write(bytes_io.read())


if __name__ == "__main__":
  app.run(main)
