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
"""Wrapper model that allows rescaling of any model's outputs."""

import importlib
import pickle
from typing import Any, Dict


import numpy as np
import robustness_metrics as rm
from sklearn import isotonic as sklearn_ir  # Needed for unpickling IR object? pylint: disable=unused-import
import tensorflow as tf
import tensorflow_probability as tfp


def create(*,
           base_model_name: str,
           base_model_path: str,
           base_model_args: str = "",
           rescaling_method: str,
           rescaling_kwargs: Dict[str, Any]):
  """Wraps a base model and re-scales its outputs.

  Usage:

  First, find the optimal rescaling parameters for your model, dataset and
  rescaling method of choice.
  For example, when using temperature scaling, find
  `beta`.
  Or for example, when using isotonic regression, run
  `isotonic_fitting(probs_from_model_output,
                    "path/to/where/regression/params/will/be/saved")`.

  Second, use this wrapper in place of your model to obtain rescaled predictions
  from your model on any dataset: Specify your model of interest as
  `base_model`, along with the desired `rescaling_method` and the
  `rescaling_kwargs` you obtained in the first step.
  For example, when using isotonic_regresion,
  use previous "path/to/where/regression/params/will/be/saved" as value to
  the key `pickle_path` in rescaling_kwargs: Dict[str, Any] with
  'rescaling_method: "isotonic_regression"'.

  Args:
   base_model_name: Name of the model to be wrapped.
   base_model_path: Import path for the base model, e.g.
     `robustness_metrics.models.bit`.
   base_model_args: Arguments (in string form) for the base model.
   rescaling_method: Name of the rescaling method.
   rescaling_kwargs: Dict with kwargs for the rescaling function.

  Returns:
    The model function and the corresponding preprocessing function.
  """
  # Name is unused, but kept in the signature so it gets stored in the database:
  del base_model_name

  # Import model:
  _, _, base_model_args = rm.common.registry.parse_name_and_kwargs(
      f"foo({base_model_args})")
  module = importlib.import_module(base_model_path)
  base_model, preprocess_fn = module.create(**base_model_args)

  if rescaling_method == "isotonic_scaling":
    with tf.io.gfile.GFile(rescaling_kwargs["pickle_path"], "rb") as handle:
      ir = pickle.load(handle)
    x_grid = np.linspace(0, 1, 10000)
    y_ref = tf.convert_to_tensor(
        ir.transform(x_grid.astype(np.float64)).reshape([1, -1]), tf.float32)

  def isotonic_scaling_fn(probs_unscaled):
    batch_size = tf.shape(probs_unscaled)[0]
    probs_scaled = tfp.math.batch_interp_regular_1d_grid(
        tf.reshape(probs_unscaled, [batch_size, -1]), 0, 1, y_ref)
    probs_scaled = tf.reshape(probs_scaled, probs_unscaled.shape)
    # Renormalize to sum to 1:
    denominator = tf.reduce_sum(probs_scaled, axis=-1, keepdims=True) + 1e-6
    probs_scaled = probs_scaled / denominator
    return probs_scaled

  def model(features):
    probs_unscaled = base_model(features)
    if rescaling_method == "none":
      return probs_unscaled
    elif rescaling_method == "temperature_scaling":
      return temperature_scaling(probs_unscaled, **rescaling_kwargs)
    elif rescaling_method == "isotonic_scaling":
      return isotonic_scaling_fn(probs_unscaled)
    else:
      raise ValueError(f"Unknown rescaling method: {rescaling_method!r}")

  # If it's a TensorFlow model, wrap it in tf.function:
  # TODO(mjlm): Check if there's a better way to test if it's a tf.function.
  if hasattr(base_model, "function_spec"):
    model = tf.function(model)

  return model, preprocess_fn


def temperature_scaling(probs, *, beta):
  logits = tf.math.log(probs + 1e-10)
  return tf.nn.softmax(beta * logits, axis=-1)







