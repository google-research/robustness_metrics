# coding=utf-8
# Copyright 2020 The Robustness Metrics Authors.
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
"""Tests for the base metrics."""

import numpy as np
import robustness_metrics as rm
import tensorflow as tf


class MetricTest(tf.test.TestCase):

  def test_registration(self):

    @rm.metrics.base.registry.register("my-metric")
    class MyMetric(rm.metrics.base.Metric):
      pass

    self.assertEqual(rm.metrics.base.get("my-metric"), MyMetric)

  def test_that_exception_is_raised_on_name_reuse(self):

    # pylint: disable=unused-variable
    @rm.metrics.base.registry.register("my-metric-1")
    class MyMetric1(rm.metrics.base.Metric):
      pass

    with self.assertRaises(ValueError):

      @rm.metrics.base.registry.register("my-metric-1")
      class MyMetric2(rm.metrics.base.Metric):
        pass

    # pylint: enable=unused-variable

  def test_that_exception_is_raised_on_unknown_metric(self):
    with self.assertRaises(KeyError):
      rm.metrics.base.get("foobar")


class AccuracyTest(tf.test.TestCase):

  def assertDictsAlmostEqual(self, dict_1, dict_2):
    self.assertEqual(dict_1.keys(), dict_2.keys())
    for key in dict_1:
      self.assertAlmostEqual(dict_1[key], dict_2[key])

  def test_single_prediction(self):
    metric = rm.metrics.base.get("accuracy")()
    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=1,
            metadata={"label": 1},
            predictions=[[.2, .8], [.7, .3]]))
    self.assertDictsAlmostEqual(metric.result(), {"accuracy": 1.})

  def test_multiple_predictions(self):
    metric = rm.metrics.base.get("accuracy")()
    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=0,
            metadata={"label": 1},
            predictions=[[.2, .8], [.7, .3]]))
    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=1, metadata={"label": 0}, predictions=[[.51, .49]]))
    self.assertDictsAlmostEqual(metric.result(), {"accuracy": 1.})

    expected_error = "(.*) but you added element id 1 twice."
    with self.assertRaisesRegex(ValueError, expected_error):
      metric.add_predictions(
          rm.common.types.ModelPredictions(
              element_id=1, metadata={"label": 0}, predictions=[[.51, .49]]))


class AggregatedAccuracyTest(tf.test.TestCase):

  def assertDictsAlmostEqual(self, dict_1, dict_2):
    self.assertEqual(dict_1.keys(), dict_2.keys())
    for key in dict_1:
      self.assertAlmostEqual(dict_1[key], dict_2[key])

  def test_multiple_predictions(self):

    def worst_case_aggregator_fn(scores):
      return np.min(scores)

    metric = rm.metrics.base.AggregatedAccuracy(
        group_element_id_field="video_frame_id",
        aggregator_fn=worst_case_aggregator_fn,
        dataset_info=None)

    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=0,
            metadata={
                "video_frame_id": "imgnet_000000/000001",
                "label": 1
            },
            predictions=[[.2, .8, .0]]))

    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=1,
            metadata={
                "video_frame_id": "imgnet_000001/000001",
                "label": 2
            },
            predictions=[[.0, .1, .9]]))

    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=2,
            metadata={
                "video_frame_id": "imgnet_000001/000002",
                "label": 0
            },
            predictions=[[.0, .1, .9]]))

    self.assertDictsAlmostEqual(metric.result(),
                                {"aggregated_accuracy": 1. / 2})

  def test_multiple_predictions_multi_label(self):

    def worst_case_aggregator_fn(scores):
      return np.min(scores)

    metric = rm.metrics.base.AggregatedAccuracy(
        group_element_id_field="video_frame_id",
        aggregator_fn=worst_case_aggregator_fn,
        dataset_info=None)

    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=0,
            metadata={
                "video_frame_id": "imgnet_000000/000001",
                "labels_multi_hot": [1, 0, 1]
            },
            predictions=[[.2, .8, .0]]))

    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=1,
            metadata={
                "video_frame_id": "imgnet_000001/000001",
                "labels_multi_hot": [0, 0, 1]
            },
            predictions=[[.0, .1, .9]]))

    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=2,
            metadata={
                "video_frame_id": "imgnet_000001/000002",
                "labels_multi_hot": [0, 0, 1]
            },
            predictions=[[.0, .1, .9]]))

    self.assertDictsAlmostEqual(metric.result(),
                                {"aggregated_accuracy": 1. / 2})

  def test_multiple_predictions_multi_label_class_filter(self):

    def worst_case_aggregator_fn(scores):
      return np.min(scores)

    metric = rm.metrics.base.AggregatedAccuracy(
        group_element_id_field="video_frame_id",
        aggregator_fn=worst_case_aggregator_fn,
        dataset_info=rm.datasets.base.DatasetInfo(
            num_classes=3,
            appearing_classes=[0, 1]))

    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=0,
            metadata={
                "video_frame_id": "imgnet_000000/000001",
                "labels_multi_hot": [1, 0, 1]
            },
            predictions=[[.8, .2, .0]]))  # Correct prediction.

    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=1,
            metadata={
                "video_frame_id": "imgnet_000001/000001",
                "labels_multi_hot": [1, 0, 1]
            },
            predictions=[[.0, .1, .9]]))  # Wrong (class 3 is filtered out).

    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=2,
            metadata={
                "video_frame_id": "imgnet_000001/000002",
                "labels_multi_hot": [1, 0, 0]
            },
            predictions=[[.2, .1, .7]]))  # Correct (class 3 filtered out).

    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=2,
            metadata={
                "video_frame_id": "imgnet_000002/000001",
                "labels_multi_hot": [0, 1, 0]
            },
            predictions=[[.1, .2, .7]]))  # Correct (class 3 filtered out).

    self.assertDictsAlmostEqual(metric.result(),
                                {"aggregated_accuracy": 2. / 3})


class PrecisionTest(tf.test.TestCase):

  def assertDictsAlmostEqual(self, dict_1, dict_2):
    self.assertEqual(dict_1.keys(), dict_2.keys())
    for key in dict_1:
      self.assertAlmostEqual(dict_1[key], dict_2[key])

  def test_single_label_top1(self):
    dataset_info = rm.datasets.base.DatasetInfo(num_classes=3)
    metric = rm.metrics.base.get("precision")(
        top_k=1, dataset_info=dataset_info)
    # Average prediction is [.45, .55, .0], so 1/1 are correct.
    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=1,
            metadata={"label": 1},
            predictions=[[.2, .8, .0], [.7, .3, .0]]))
    self.assertDictsAlmostEqual(metric.result(), {"precision@1": 1})

  def test_single_label_top2(self):
    dataset_info = rm.datasets.base.DatasetInfo(num_classes=3)
    metric = rm.metrics.base.get("precision")(
        top_k=2, dataset_info=dataset_info)
    # Average prediction is [.45, .4, .15], so 1/2 are correct.
    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=1,
            metadata={"label": 1},
            predictions=[
                [.2, .8, .0],
                [.7, .0, .3],
            ]))
    # Average prediction is [.0, .525, .475], so 1/2 are correct.
    self.assertDictsAlmostEqual(metric.result(), {"precision@2": 1 / 2})
    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=2,
            metadata={"label": 0},
            predictions=[
                [.0, .95, .05],
                [.0, .1, .9],
            ]))
    # Average accuracy over the above predictions is (1/2 + 1/2) / 2 = 1/4.
    self.assertDictsAlmostEqual(metric.result(), {"precision@2": 1 / 4})

  def test_multi_label_top2(self):
    dataset_info = rm.datasets.base.DatasetInfo(num_classes=3)
    metric = rm.metrics.base.get("precision")(
        top_k=2, dataset_info=dataset_info)
    # Average prediction is [.1, .75, .15], so 2/2 are correct.
    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=1,
            metadata={"labels_multi_hot": [0, 1, 1]},
            predictions=[
                [.2, .8, .0],
                [.0, .7, .3],
            ]))
    self.assertDictsAlmostEqual(metric.result(), {"precision@2": 1})
    # Average prediction is [.0, .525, .475], so 1/2 are correct.
    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=2,
            metadata={"labels_multi_hot": [1, 1, 0]},
            predictions=[
                [.0, .95, .05],
                [.0, .1, .9],
            ]))
    # Average accuracy over the above predictions is (1 + 1/2) / 2 = 3/4.
    self.assertDictsAlmostEqual(metric.result(), {"precision@2": 3 / 4})


class MultiLabelTopKTest(tf.test.TestCase):

  def assertDictsAlmostEqual(self, dict_1, dict_2):
    self.assertEqual(dict_1.keys(), dict_2.keys())
    for key in dict_1:
      self.assertAlmostEqual(dict_1[key], dict_2[key])

  def test_top1(self):
    dataset_info = rm.datasets.base.DatasetInfo(num_classes=3)
    metric = rm.metrics.base.get("accuracy_top_k")(
        top_k=1, dataset_info=dataset_info)
    # Average prediction is [.45, .55, .0], so 0/1 are correct.
    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=1,
            metadata={"labels_multi_hot": [1, 0, 0]},
            predictions=[[.2, .8, .0], [.7, .3, .0]]))
    self.assertDictsAlmostEqual(metric.result(), {"accuracy@1": 0})
    # If we also set the second position to be true, we get a correct guess.
    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=1,
            metadata={"labels_multi_hot": [1, 1, 0]},
            predictions=[[.2, .8, .0], [.7, .3, .0]]))
    # The average of the above predictions is 0.5.
    self.assertDictsAlmostEqual(metric.result(), {"accuracy@1": 0.5})

  def test_single_label_top2(self):
    dataset_info = rm.datasets.base.DatasetInfo(num_classes=3)
    metric = rm.metrics.base.get("accuracy_top_k")(
        top_k=2, dataset_info=dataset_info)
    # Average prediction is [.45, .4, .15], so we get a wrong guess.
    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=1,
            metadata={"labels_multi_hot": [0, 0, 1]},
            predictions=[
                [.2, .8, .0],
                [.7, .0, .3],
            ]))
    self.assertDictsAlmostEqual(metric.result(), {"accuracy@2": 0})
    # Average prediction is [.0, .525, .475], so we get a correct guess.
    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=2,
            metadata={"labels_multi_hot": [0, 0, 1]},
            predictions=[
                [.0, .95, .05],
                [.0, .1, .9],
            ]))
    # Average accuracy over the above predictions is 1/2.
    self.assertDictsAlmostEqual(metric.result(), {"accuracy@2": 1 / 2})
    # Average prediction is [.0, .525, .475], so we get a correct guess.
    metric.add_predictions(
        rm.common.types.ModelPredictions(
            element_id=2,
            metadata={"labels_multi_hot": [1, 0, 1]},
            predictions=[
                [.0, .95, .05],
                [.0, .1, .9],
            ]))
    # Average accuracy over the above three predictions is 2/3.
    self.assertDictsAlmostEqual(metric.result(), {"accuracy@2": 2 / 3})


if __name__ == "__main__":
  tf.test.main()
