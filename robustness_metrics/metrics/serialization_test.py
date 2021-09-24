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

"""Tests for serialization."""

import numpy as np
import robustness_metrics as rm
import tensorflow as tf


class SerializationTest(tf.test.TestCase):

  def testSerializer(self):
    path = self.create_tempdir().create_file('myfile.tfrecords').full_path
    serializer = rm.metrics.Serializer(path)
    predictions_and_metadata = [
        (np.array([0., 1., 2.], dtype=np.float32),
         {'element_id': 1, 'label': 1}),
        (np.array([3., 4., 5.], dtype=np.float32),
         {'element_id': 2, 'label': 2}),
        (np.array([6., 7., 8.], dtype=np.float32),
         {'element_id': 3, 'label': 3}),
    ]

    serializer.add_predictions(*predictions_and_metadata[0])
    serializer.add_predictions(*predictions_and_metadata[1])
    serializer.flush()
    actual = list(serializer.read_predictions())
    for x, y in zip(predictions_and_metadata[:2], actual):
      self.assertAllEqual(x[0], y[0])
      self.assertEqual(x[1], y[1])

    serializer.add_predictions(*predictions_and_metadata[2])
    serializer.flush()
    actual = list(serializer.read_predictions())
    for x, y in zip(predictions_and_metadata, actual):
      self.assertAllEqual(x[0], y[0])
      self.assertEqual(x[1], y[1])

if __name__ == '__main__':
  tf.test.main()
