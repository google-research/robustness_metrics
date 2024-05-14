# coding=utf-8
# Copyright 2024 The Robustness Metrics Authors.
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

"""Tests for the retrieval metrics."""

from absl.testing import parameterized
import robustness_metrics as rm
import tensorflow as tf


class AucPrTest(tf.test.TestCase):

  def assertDictsAlmostEqual(self, dict_1, dict_2):
    self.assertEqual(dict_1.keys(), dict_2.keys())
    for key in dict_1:
      self.assertAlmostEqual(dict_1[key], dict_2[key])

  def test_auc_pr_binary(self):
    # Test case inspired by the example from:
    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    metric = rm.metrics.get("auc_pr(is_binary_classification=True)")
    metric.add_predictions(
        rm.common.types.ModelPredictions(predictions=[[.9, .1]]),
        metadata={
            "label": 0,
            "element_id": 0
        })
    metric.add_predictions(
        rm.common.types.ModelPredictions(predictions=[[.6, .4]]),
        metadata={
            "label": 0,
            "element_id": 1
        })
    metric.add_predictions(
        rm.common.types.ModelPredictions(predictions=[[.65, .35]]),
        metadata={
            "label": 1,
            "element_id": 2
        })
    metric.add_predictions(
        rm.common.types.ModelPredictions(predictions=[[0.2, 0.8]]),
        metadata={
            "label": 1,
            "element_id": 3
        })
    self.assertDictsAlmostEqual(metric.result(), {"auc_pr": 0.83333333})

  def test_auc_pr_for_ood(self):
    # Test case inspired by the example from:
    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    metric = rm.metrics.get("auc_pr(is_binary_classification=False)")
    metric.add_predictions(
        # We swap some predictions, which does not affect the results because of
        # of the application of max(., axis=1).
        rm.common.types.ModelPredictions(predictions=[[0.1, 0.01]]),
        metadata={
            "label": 0,
            "element_id": 0
        })
    metric.add_predictions(
        # We swap some predictions, which does not affect the results because of
        # of the application of max(., axis=1).
        rm.common.types.ModelPredictions(predictions=[[.4, .006]]),
        metadata={
            "label": 0,
            "element_id": 1
        })
    metric.add_predictions(
        rm.common.types.ModelPredictions(predictions=[[.05, .35]]),
        metadata={
            "label": 1,
            "element_id": 2
        })
    metric.add_predictions(
        rm.common.types.ModelPredictions(predictions=[[0.2, 0.8]]),
        metadata={
            "label": 1,
            "element_id": 3
        })
    self.assertDictsAlmostEqual(metric.result(), {"auc_pr": 0.83333333})


class AucRocAndFalsePositiveRate95Test(parameterized.TestCase,
                                       tf.test.TestCase):

  @parameterized.parameters(
      ("auc_roc", 0.25),
      ("fpr95", 1.0),
  )
  def test_auc_roc_and_fpr95_binary(self, metric_name, result):
    metric = rm.metrics.get(f"{metric_name}(is_binary_classification=True)")
    metric.add_predictions(
        rm.common.types.ModelPredictions(predictions=[[.8, .2]]),
        metadata={
            "label": 0,
            "element_id": 0
        })
    metric.add_predictions(
        rm.common.types.ModelPredictions(predictions=[[.6, .4]]),
        metadata={
            "label": 0,
            "element_id": 1
        })
    metric.add_predictions(
        rm.common.types.ModelPredictions(predictions=[[.65, .35]]),
        metadata={
            "label": 1,
            "element_id": 2
        })
    metric.add_predictions(
        rm.common.types.ModelPredictions(predictions=[[0.9, 0.1]]),
        metadata={
            "label": 1,
            "element_id": 3
        })
    self.assertDictEqual(metric.result(), {metric_name: result})


class MspTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ("auc_roc",),
      ("fpr95",),
      ("auc_pr",),
  )
  def test_setting_one_minus_msp(self, metric_name):
    msp_metric = rm.metrics.get(f"{metric_name}(one_minus_msp=False)")
    msp_metric.add_predictions(
        rm.common.types.ModelPredictions(predictions=[[.7, .1, .2]]),
        metadata={
            "label": 0,
            "element_id": 0
        })
    msp_metric.add_predictions(
        rm.common.types.ModelPredictions(predictions=[[.2, .5, .3]]),
        metadata={
            "label": 1,
            "element_id": 1
        })

    one_minus_msp_metric = rm.metrics.get(f"{metric_name}(one_minus_msp=True)")
    one_minus_msp_metric.add_predictions(
        # msp = 0.3 so that 1 - msp = 0.7, as above.
        rm.common.types.ModelPredictions(predictions=[[.1, .3, .2]]),
        metadata={
            "label": 0,
            "element_id": 0
        })
    one_minus_msp_metric.add_predictions(
        # msp = 0.5 so that 1 - msp = 0.5, as above.
        rm.common.types.ModelPredictions(predictions=[[.5, .0, .3]]),
        metadata={
            "label": 1,
            "element_id": 1
        })
    self.assertAllClose(msp_metric.result(), one_minus_msp_metric.result())


if __name__ == "__main__":
  tf.test.main()
