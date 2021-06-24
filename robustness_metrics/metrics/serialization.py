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
"""Metrics that take predictions and serializes them."""

import json
from typing import Iterator, Tuple
from robustness_metrics.common import types
from robustness_metrics.metrics import base as metrics_base
import tensorflow as tf


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


@metrics_base.registry.register("serializer")
class Serializer(metrics_base.Metric):
  """Serializes the predictions to disk in TFRecord format."""

  def __init__(self, path: str, dataset_info=None):
    super().__init__(dataset_info)
    self._path = path
    self._record_writer = None

  def add_predictions(self,
                      model_predictions: types.ModelPredictions,
                      metadata: types.Features) -> None:
    if self._record_writer is None:
      self._record_writer = tf.io.TFRecordWriter(self._path)
    serialized_predictions = tf.io.serialize_tensor(
        tf.convert_to_tensor(model_predictions.predictions, dtype=tf.float32))
    serialized_metadata = {}
    for key, value in metadata.items():
      if isinstance(value, tf.Tensor):
        value = value.numpy()
      try:
        value = value.tolist()
      except AttributeError:
        pass
      # Convert bytes (e.g., ImageNetVidRobust's video_frame_id).
      if isinstance(value, bytes):
        value = value.decode("utf-8")
      serialized_metadata[key] = value
    serialized_metadata = json.dumps(serialized_metadata).encode()
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        "predictions": _bytes_feature(serialized_predictions.numpy()),
        "metadata": _bytes_feature(serialized_metadata),
    }))
    self._record_writer.write(tf_example.SerializeToString())

  def result(self):
    return {"path": self._path}

  def flush(self):
    """Flush the file."""
    if self._record_writer is not None:
      return self._record_writer.flush()

  def read_predictions(
      self
  ) -> Iterator[Tuple[types.ModelPredictions, types.Features]]:
    """Reads path in order to yield each prediction and metadata."""

    def parse(features_serialized):
      features = {
          "predictions": tf.io.FixedLenFeature([], tf.string),
          "metadata": tf.io.FixedLenFeature([], tf.string),
      }
      features = tf.io.parse_single_example(features_serialized, features)
      features["predictions"] = tf.io.parse_tensor(
          features["predictions"], tf.float32)
      return features

    path = tf.convert_to_tensor(self._path)
    dataset = tf.data.TFRecordDataset(path).map(parse)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA)
    dataset = dataset.with_options(options)
    for example in dataset:
      prediction = types.ModelPredictions(
          predictions=example["predictions"].numpy())
      metadata = json.loads(example["metadata"].numpy())
      # Apply a special case to lists of size 1. We need to adjust for the fact
      # that int-casting a Tensor with shape [1] works (this may be the original
      # element), but int-casting a list of size 1 (this may be the saved
      # element) doesn't work.
      for key, value in metadata.items():
        if isinstance(value, list) and len(value) == 1:
          metadata[key] = value[0]
      yield prediction, metadata
