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

"""Tests for robustness_metrics.models.rescaling."""


import importlib
import os
from typing import Any, Dict

from absl.testing import absltest
import numpy as np
import robustness_metrics as rm
from robustness_metrics.bin import common as bin_common
from scipy import special
import tensorflow as tf

MOCK_OUTPUT_PROBS = np.array([0.1, 0.9])
MOCK_OUTPUT_PROBS_ISOTONIC_1 = np.array([1., 0.])
MOCK_OUTPUT_PROBS_ISOTONIC_2 = np.array([0., 1.])
BATCH_SIZE = 1000
# Batch size /equiv num_of_classes(uniform_imagenet_tensorflow)


class MockModel:

  def __call__(self, features: Dict[Any, Any]):
    del features
    return MOCK_OUTPUT_PROBS[np.newaxis, ...]


class RescalingTest(absltest.TestCase):

  def test_rescaling_model(self):
    """Tests that the rescaling model wrapper works end-to-end."""

    # Settings for the model to be calibrated:
    base_model_path = "robustness_metrics.models.uniform_imagenet_tensorflow"
    rescaling_method = "temperature_scaling"
    model_args = ("base_model_name='uniform_imagenet_tensorflow',"
                  f"base_model_path={base_model_path!r},"
                  f"rescaling_method={rescaling_method!r},"
                  "rescaling_kwargs={'beta':1.0},")
    _, _, model_kwargs = rm.common.registry.parse_name_and_kwargs(
        f"foo({model_args})")

    model_dir = os.path.normpath(os.path.dirname(__file__))
    model_path = os.path.join(model_dir, "rescaling.py")

    module = rm.bin.common.load_module_from_path(model_path)
    wrapped_model, _ = module.create(**model_kwargs)

    module = importlib.import_module(base_model_path)
    base_model, _ = module.create()

    features = {"image": tf.ones((16, 224, 224, 3))}

    preds_wrapped = wrapped_model(features)
    preds_base = base_model(features)

    self.assertIsInstance(preds_wrapped, tf.Tensor)
    self.assertEqual(preds_base.shape, preds_wrapped.shape)
    np.testing.assert_array_almost_equal(
        preds_base.numpy(), preds_wrapped.numpy())

  def test_temperature_scaling(self):
    # Settings for the model to be calibrated:
    base_model_path = "robustness_metrics.models.rescaling_test"
    rescaling_method = "temperature_scaling"
    model_args = ("base_model_name='mock_model',"
                  f"base_model_path={base_model_path!r},"
                  f"rescaling_method={rescaling_method!r},"
                  "rescaling_kwargs={'beta':1.0},")

    model_dir = os.path.normpath(os.path.dirname(__file__))
    model_path = os.path.join(model_dir, "rescaling.py")
    module = bin_common.load_module_from_path(model_path)
    _, _, kwargs = rm.common.registry.parse_name_and_kwargs(
        f"foo({model_args})")

    # Model 1 uses beta=1.0:
    model1, _ = module.create(**kwargs)
    preds1 = model1(features={})
    np.testing.assert_allclose(preds1[0], MOCK_OUTPUT_PROBS)

    # Model 2 uses beta=2.0:
    kwargs["rescaling_kwargs"] = {"beta": 2.0}
    model2, _ = module.create(**kwargs)
    preds2 = model2(features={})
    np.testing.assert_allclose(
        preds2[0], special.softmax(np.log(MOCK_OUTPUT_PROBS) * 2.0))


def create():
  return MockModel(), None


if __name__ == "__main__":
  absltest.main()
