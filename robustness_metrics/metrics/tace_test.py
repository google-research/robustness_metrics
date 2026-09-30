# coding=utf-8
# Copyright 2026 The Robustness Metrics Authors.
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
"""Thresholded adaptive calibration uses the requested threshold."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import robustness_metrics as rm


_PROBS = np.array([[.99, .005, .005], [.1, .8, .1],
                   [.2, .2, .6], [.05, .1, .85]])
_LABELS = np.array([1, 1, 0, 2])


def _one_bin_error(threshold):
  # With one bin per class, calibration is the absolute difference of
  # empirical accuracy and mean confidence among the retained probabilities.
  errors = []
  for class_id in range(_PROBS.shape[1]):
    keep = _PROBS[:, class_id] > threshold
    if np.any(keep):
      errors.append(abs(np.mean(_LABELS[keep] == class_id)
                        - np.mean(_PROBS[keep, class_id])))
    else:
      errors.append(0.)
  return np.mean(errors)


class ThresholdedAdaptiveCalibrationErrorTest(parameterized.TestCase):

  @parameterized.product(
      threshold=[0., .01, .1, .5, .99, 1.], registry=[False, True])
  def test_one_bin_result_matches_filtered_class_means(
      self, threshold, registry):
    if registry:
      metric = rm.metrics.get(f'tace(num_bins=1,threshold={threshold})')
    else:
      metric = rm.metrics.ThresholdedAdaptiveCalibrationError(
          num_bins=1, threshold=threshold)
    metric.add_batch(_PROBS, label=_LABELS)
    self.assertAlmostEqual(metric.result()['gce'], _one_bin_error(threshold))

  @parameterized.parameters('tace', 'tace(threshold=0.01)')
  def test_default_threshold_differs_from_ace(self, spec):
    metric = rm.metrics.get(spec)
    equivalent = rm.metrics.GeneralCalibrationError(
        None, binning_scheme='adaptive', max_prob=False, class_conditional=True,
        norm='l1', num_bins=30, threshold=.01)
    ace = rm.metrics.AdaptiveCalibrationError()
    for item in [metric, equivalent, ace]:
      item.add_batch(_PROBS, label=_LABELS)
    self.assertAlmostEqual(metric.result()['gce'], equivalent.result()['gce'])
    self.assertNotAlmostEqual(metric.result()['gce'], ace.result()['gce'])

  @parameterized.parameters(1, 2, 7, 30)
  def test_explicit_zero_matches_ace(self, num_bins):
    metric = rm.metrics.ThresholdedAdaptiveCalibrationError(
        num_bins=num_bins, threshold=0.)
    ace = rm.metrics.AdaptiveCalibrationError(num_bins=num_bins)
    for item in [metric, ace]:
      item.add_batch(_PROBS, label=_LABELS)
    self.assertAlmostEqual(metric.result()['gce'], ace.result()['gce'])

  @parameterized.parameters(1, 2, 7, 30)
  def test_threshold_and_bin_count_match_general_calibration(self, num_bins):
    metric = rm.metrics.ThresholdedAdaptiveCalibrationError(
        num_bins=num_bins, threshold=.1)
    equivalent = rm.metrics.GeneralCalibrationError(
        None, binning_scheme='adaptive', max_prob=False, class_conditional=True,
        norm='l1', num_bins=num_bins, threshold=.1)
    for item in [metric, equivalent]:
      item.add_batch(_PROBS, label=_LABELS)
    self.assertAlmostEqual(metric.result()['gce'], equivalent.result()['gce'])

  def test_individual_predictions_match_batch_result(self):
    metric = rm.metrics.ThresholdedAdaptiveCalibrationError(
        num_bins=1, threshold=.1)
    for index, (probabilities, label) in enumerate(zip(_PROBS, _LABELS)):
      metric.add_predictions(
          rm.common.types.ModelPredictions(predictions=[probabilities]),
          {'label': label, 'element_id': index})
    self.assertAlmostEqual(metric.result()['gce'], _one_bin_error(.1))
    self.assertAlmostEqual(metric.result()['gce'], _one_bin_error(.1))


if __name__ == '__main__':
  absltest.main()
