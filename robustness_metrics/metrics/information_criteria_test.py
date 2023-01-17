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

"""Tests for information criteria."""

import robustness_metrics as rm
import tensorflow.compat.v1 as tf


class InformationCriteriaTest(tf.test.TestCase):

  def testEnsembleCrossEntropy(self):
    """Checks that ensemble cross entropy lower-bounds Gibbs cross entropy."""
    # For multi-class classifications.
    batch_size = 2
    num_classes = 3
    ensemble_size = 5
    labels = tf.random.uniform(
        [batch_size], minval=0, maxval=num_classes, dtype=tf.int32)
    logits = tf.random.normal([ensemble_size, batch_size, num_classes])
    ensemble_error = rm.metrics.EnsembleCrossEntropy()
    ensemble_error.add_batch(labels, logits)
    ensemble_error = list(ensemble_error.result().values.values())[0]
    gibbs_error = rm.metrics.GibbsCrossEntropy()
    gibbs_error.add_batch(labels, logits)
    gibbs_error = list(gibbs_error.result().values())[0]
    self.assertEqual(ensemble_error.shape, ())
    self.assertEqual(gibbs_error.shape, ())
    self.assertLessEqual(ensemble_error, gibbs_error)

    # For binary classifications.
    num_classes = 1
    labels = tf.random.uniform(
        [batch_size], minval=0, maxval=num_classes, dtype=tf.float32)
    logits = tf.random.normal([ensemble_size, batch_size, num_classes])
    loss_logits = tf.squeeze(logits, axis=-1)
    ensemble_error = rm.metrics.EnsembleCrossEntropy(binary=True)
    ensemble_error.add_batch(labels, loss_logits)
    ensemble_error = list(ensemble_error.result().values())[0]
    gibbs_error = rm.metrics.GibbsCrossEntropy(binary=True)
    gibbs_error.add_batch(labels, loss_logits)
    gibbs_error = list(gibbs_error.result().values())[0]
    self.assertEqual(ensemble_error.shape, ())
    self.assertEqual(gibbs_error.shape, ())
    self.assertLessEqual(ensemble_error, gibbs_error)
