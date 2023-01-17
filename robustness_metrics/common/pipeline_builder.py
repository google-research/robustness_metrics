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

"""Image preprocessing library."""
from robustness_metrics.common import ops
from robustness_metrics.common import registry
import tensorflow.compat.v1 as tf


TPU_SUPPORTED_DTYPES = [
    tf.bool, tf.int32, tf.int64, tf.bfloat16, tf.float32, tf.complex64,
    tf.uint32
]


def _get_delete_field(key):
  def _delete_field(datum):
    if key in datum:
      del datum[key]
    return datum
  return _delete_field


def keep_only_tpu_types(data):
  """Remove data that are TPU-incompatible (e.g. filename of type tf.string)."""
  for key in list(data.keys()):
    if isinstance(data[key], dict):
      data[key] = keep_only_tpu_types(data[key])
    else:
      if data[key].dtype not in TPU_SUPPORTED_DTYPES:
        tf.logging.warning(
            "Removing key '{}' from data dict because its dtype {} is not in "
            " the supported dtypes: {}".format(key, data[key].dtype,
                                               TPU_SUPPORTED_DTYPES))
        data = _get_delete_field(key)(data)
  return data


def get_preprocess_fn(pp_pipeline, remove_tpu_dtypes=True):
  """Transform an input string into the preprocessing function.

  The minilanguage is as follows:

    fn1|fn2(arg, arg2,...)|...

  And describes the successive application of the various `fn`s to the input,
  where each function can optionally have one or more arguments, which are
  either positional or key/value, as dictated by the `fn`.

  The output preprocessing function expects a dictinary as input. This
  dictionary should have a key "image" that corresponds to a 3D tensor
  (height x width x channel).

  Args:
    pp_pipeline: A string describing the pre-processing pipeline.
    remove_tpu_dtypes: Whether to remove TPU incompatible types of data.

  Returns:
    preprocessing function.

  Raises:
    ValueError: if preprocessing function name is unknown
  """
  def _preprocess_fn(data):
    """The preprocessing function that is returned."""

    # Validate input
    if not isinstance(data, dict):
      raise ValueError("Argument `data` must be a dictionary, "
                       "not %s" % str(type(data)))

    # Apply all the individual steps in sequence.
    tf.logging.info("Data before pre-processing:\n%s", data)
    for lookup_string in pp_pipeline.split("|"):
      # These calls are purely positional, so no need for kwargs.
      name, args, _ = registry.parse_name_and_kwargs(lookup_string)
      cls = ops.get(name)
      data = cls.apply(*args)(data)

    if remove_tpu_dtypes:
      data = keep_only_tpu_types(data)
    tf.logging.info("Data after pre-processing:\n%s", data)
    return data

  return _preprocess_fn
