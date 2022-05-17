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

"""Tests for robustness_metrics.metrics.diversity."""

from absl.testing import parameterized
import robustness_metrics as rm
import tensorflow as tf


class DiversityTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (1, 5),
      (2, 3),
  )
  def test_bregman_kl_variance(self, batch_size, num_classes):
    logits_1 = tf.random.normal((batch_size, num_classes))
    logits_2 = tf.random.normal((batch_size, num_classes))

    prediction = tf.nn.softmax(tf.stack([logits_1, logits_2], axis=0))

    actual_variance = rm.metrics.diversity.bregman_kl_variance(prediction)
    self.assertEqual(actual_variance.shape, [batch_size])

    central_pred = tf.nn.softmax(.5 * (logits_1 + logits_2))
    kl_1 = tf.keras.losses.kl_divergence(central_pred, tf.nn.softmax(logits_1))
    kl_2 = tf.keras.losses.kl_divergence(central_pred, tf.nn.softmax(logits_2))

    self.assertAllClose(actual_variance, .5 * (kl_1 + kl_2))

  @parameterized.parameters(
      (rm.metrics.diversity.cosine_distance,),
      (rm.metrics.diversity.disagreement,),
      (rm.metrics.diversity.kl_divergence,),
  )
  def testDiversityMetrics(self, metric_fn):
    # TODO(trandustin): Test shapes. Need to change API to make it consistent.
    batch_size = 2
    num_classes = 3
    logits_one = tf.random.normal([batch_size, num_classes])
    logits_two = tf.random.normal([batch_size, num_classes])
    _ = metric_fn(logits_one, logits_two)

  def testAveragePairwiseDiversity(self):
    num_models = 3
    num_classes = 5
    diversity = rm.metrics.AveragePairwiseDiversity()

    batch_size = 2
    logits = tf.random.normal([num_models, batch_size, num_classes])
    probs = tf.nn.softmax(logits)
    diversity.add_batch(probs)

    batch_size = 3
    logits = tf.random.normal([num_models, batch_size, num_classes])
    probs = tf.nn.softmax(logits)
    diversity.add_batch(probs)

    results = diversity.result()
    self.assertLen(results, 4)
    self.assertIsInstance(results['disagreement'], float)
    self.assertIsInstance(results['average_kl'], float)
    self.assertIsInstance(results['cosine_similarity'], float)
    self.assertIsInstance(results['bregman_kl_variance'], float)

  def testAveragePairwiseDiversityWithoutUpdate(self):
    diversity = rm.metrics.AveragePairwiseDiversity()
    results = diversity.result()
    self.assertLen(results, 4)
    self.assertEqual(results['disagreement'], 0.0)
    self.assertEqual(results['average_kl'], 0.0)
    self.assertEqual(results['cosine_similarity'], 0.0)
    self.assertEqual(results['bregman_kl_variance'], 0.0)


if __name__ == '__main__':
  tf.test.main()
