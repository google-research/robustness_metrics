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

"""Tests for builder."""
from absl.testing import absltest
import numpy as np
from robustness_metrics.common import pipeline_builder
import tensorflow as tf


class PipelineBuilderTest(absltest.TestCase):

  def test_preprocessing_pipeline(self):
    pp_str = ("pad(4)|pad((4,4))|inception_crop|resize(256)|resize((256, 256))|"
              "random_crop(240)|central_crop((80, 120))|"
              "flip_lr|flip_ud|standardize(0, 1)|value_range(0,1)|"
              "value_range(-1,1)")
    pp_fn = pipeline_builder.get_preprocess_fn(pp_str)

    # Typical image input
    x = tf.Variable(np.random.randint(0, 256, [640, 480, 3]))

    result = pp_fn({"image": x})
    image = result["image"].numpy()
    self.assertEqual(image.shape, (80, 120, 3))
    self.assertLessEqual(np.max(image), 1)
    self.assertGreaterEqual(np.min(image), -1)

  def test_batched_preprocessing_pipeline(self):
    pp_str = ("pad(4)|pad((4,4))|replicate(4)|inception_crop(300)|resize(256)|"
              "resize((256, 256))|random_crop(240)|"
              "central_crop((80, 120))|flip_lr|flip_ud|standardize(0, 1)|"
              "value_range(0,1)|value_range(-1,1)")
    pp_fn = pipeline_builder.get_preprocess_fn(pp_str)

    # Typical image input
    x = tf.Variable(np.random.randint(0, 256, [640, 480, 3]))

    result = pp_fn({"image": x})
    image = result["image"].numpy()
    self.assertEqual(image.shape, (4, 80, 120, 3))
    self.assertLessEqual(np.max(image), 1)
    self.assertGreaterEqual(np.min(image), -1)

  def test_num_args_exception(self):

    x = tf.Variable(np.random.randint(0, 256, [224, 224, 3]), dtype=tf.uint8)
    for pp_str in [
        "inception_crop(1)",
        "resize()",
        "resize(1, 1, 1)"
        "flip_lr(1)",
        "central_crop()",
    ]:
      with self.assertRaises(ValueError):
        pipeline_builder.get_preprocess_fn(pp_str)(x)

  def test_preprocessing_pipeline_multi_channel(self):
    pp_str = ("resize((256,256))|random_crop((224,224))|"
              "value_range_mc(-1,1,0,0,0,0,999,999,999,999)|"
              "select_channels([0,1,3])")
    pp_fn = pipeline_builder.get_preprocess_fn(pp_str)
    # Example multi-channel input
    x = tf.Variable(np.random.randint(0, 999, [64, 64, 4]))
    result = pp_fn({"image": x})
    image = result["image"].numpy()
    self.assertEqual(image.shape, (224, 224, 3))
    self.assertLessEqual(np.max(image), 1)
    self.assertGreaterEqual(np.min(image), -1)


if __name__ == "__main__":
  absltest.main()
