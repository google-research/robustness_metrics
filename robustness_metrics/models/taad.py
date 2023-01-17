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

"""Wraps models trained with the task_adaptation library."""
from robustness_metrics.common import pipeline_builder
import tensorflow as tf
import tensorflow_hub as hub


def create(hub_path: str,
           preprocess_config: str = None,
           signature: str = "representation",
           logits_key: str = "logits"):
  """Returns a model using the specified hub signature and preprocessing."""
  module = hub.KerasLayer(hub_path, signature=signature, output_key=logits_key)

  @tf.function
  def model(features):
    return tf.nn.softmax(module(features["image"]), axis=-1)

  if preprocess_config:
    preprocess_fn = pipeline_builder.get_preprocess_fn(
        preprocess_config, remove_tpu_dtypes=False)
  else:
    preprocess_fn = None

  return model, preprocess_fn
