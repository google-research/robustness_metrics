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
"""Tests for uncertainty metrics."""
import itertools
import math

from absl import logging
from absl.testing import parameterized
import numpy as np
import robustness_metrics as rm
import sklearn.model_selection
import tensorflow as tf
import tensorflow_probability as tfp


_GCE_DEFAULT = ("gce(binning_scheme='adaptive',max_prob=True,"
                "class_conditional=False,norm='l2',num_bins=2,threshold=0.0)")
_GCE_EXPLICIT_DEFAULT = ("gce(binning_scheme='adaptive',max_prob=True,"
                         "class_conditional=False,norm='l2',num_bins=2,"
                         "threshold=0.0,recalibration_method=None)")
_GCE_TEMP_SCALING_ALL = (
    "gce(binning_scheme='adaptive',max_prob=True,"
    "class_conditional=False,norm='l2',num_bins=2,threshold=0.0,"
    "recalibration_method='temperature_scaling')")
_GCE_ISOTONIC_REGR_ALL = (
    "gce(binning_scheme='adaptive',max_prob=True,"
    "class_conditional=False,norm='l2',num_bins=2,threshold=0.0,"
    "recalibration_method='isotonic_regression')")
_GCE_TEMP_SCALING_SPLIT = (
    "gce(binning_scheme='adaptive',max_prob=True,"
    "class_conditional=False,norm='l2',num_bins=2,"
    "threshold=0.0,recalibration_method='temperature_scaling',"
    "fit_on_percent=60.0,seed=3765486)")
_GCE_ISOTONIC_REGR_SPLIT = ("gce(binning_scheme='adaptive',max_prob=True,"
                            "class_conditional=False,norm='l2',num_bins=2,"
                            "threshold=0.0,recalibration_method="
                            "'isotonic_regression',fit_on_percent=60.0,"
                            "seed=3765486)")

_UNCERTAINTY_METRICS = [
    "ece", "nll", "brier", _GCE_DEFAULT, _GCE_EXPLICIT_DEFAULT,
    _GCE_TEMP_SCALING_ALL, _GCE_ISOTONIC_REGR_ALL, _GCE_TEMP_SCALING_SPLIT,
    _GCE_ISOTONIC_REGR_SPLIT, "temperature_scaling"]


def _get_info(num_classes=2):
  return rm.datasets.base.DatasetInfo(num_classes=num_classes)


def _normalize(x):
  return [x_i / sum(x) for x_i in x]


def _with_labelset(name):
  if "(" in name:
    return name[:-1] +  ",use_dataset_labelset=True)"
  else:
    return f"{name}(use_dataset_labelset=True)"


def _with_tempdir(name, tempdir):
  if "(" in name:
    return name[:-1] +  f",pickle_path='{tempdir}')"
  else:
    return f"{name}(pickle_path='{tempdir}')"


class KerasMetricTest(parameterized.TestCase, tf.test.TestCase):

  def assertDictsAlmostEqual(self, dict_1, dict_2):
    self.assertEqual(dict_1.keys(), dict_2.keys())
    for key in dict_1:
      self.assertAlmostEqual(dict_1[key], dict_2[key], places=5)

  @parameterized.parameters([(name,) for name in _UNCERTAINTY_METRICS])
  def test_binary_prediction_two_predictions_per_element(self, name):
    if "gce" == name[:3]:
      tempdir = self.create_tempdir().full_path
      # Need unmodified `name` to look up expected output.
      tempname = _with_tempdir(name, tempdir)
    else:
      tempname = name
    metric = rm.metrics.get(tempname, _get_info(2))
    metric_ls = rm.metrics.get(
        _with_labelset(tempname),
        rm.datasets.base.DatasetInfo(num_classes=3, appearing_classes=[0, 1]))
    metric.add_predictions(
        rm.common.types.ModelPredictions(predictions=[[.2, .8], [.7, .3]]),
        metadata={"label": 1, "element_id": 1})
    metric_ls.add_predictions(
        rm.common.types.ModelPredictions(
            predictions=[_normalize([.2, .8, .3]),
                         _normalize([.7, .3, .5])]),
        metadata={"label": 1, "element_id": 1})
    expected_output = {
        _GCE_DEFAULT: {"gce": .45},
        _GCE_EXPLICIT_DEFAULT: {"gce": 0.45},
        _GCE_TEMP_SCALING_ALL: {"gce": 0.0, "beta": 1.6040830888247386e+33},
        _GCE_ISOTONIC_REGR_ALL: {"gce": 0.0},
        _GCE_TEMP_SCALING_SPLIT: {"gce": 0.0},
        _GCE_ISOTONIC_REGR_SPLIT: {"gce": 0.0},
        "ece": {"ece": .45},
        "nll": {"nll": -math.log((.8 + .3) / 2)},
        "brier": {"brier": ((.2 + .7) / 2)**2 + ((.2 + .7) / 2)**2},
        "temperature_scaling": {"beta": 1.6040830888247386e+33},
    }[name]
    if name == _GCE_TEMP_SCALING_SPLIT:
      self.assertAlmostEqual(metric.result()["gce"],
                             expected_output["gce"])
      self.assertAllGreater(metric.result()["beta"], 0)
      self.assertAlmostEqual(metric_ls.result()["gce"],
                             expected_output["gce"])
      self.assertAllGreater(metric_ls.result()["beta"], 0)
    else:
      self.assertDictsAlmostEqual(metric.result(), expected_output)
      self.assertDictsAlmostEqual(metric_ls.result(), expected_output)

  @parameterized.parameters([(name,) for name in _UNCERTAINTY_METRICS])
  def test_binary_predictions_on_different_predictions(self, name):
    if "gce" == name[:3]:
      tempdir = self.create_tempdir().full_path
      # Need unmodified `name` to look up expected output.
      tempname = _with_tempdir(name, tempdir)
    else:
      tempname = name
    metric = rm.metrics.get(tempname, _get_info(2))
    metric_ls = rm.metrics.get(
        _with_labelset(tempname),
        rm.datasets.base.DatasetInfo(num_classes=3, appearing_classes=[0, 2]))
    metric.add_predictions(
        rm.common.types.ModelPredictions(predictions=[[.2, .8]]),
        metadata={"label": 1, "element_id": 1})
    metric_ls.add_predictions(
        rm.common.types.ModelPredictions(
            predictions=[_normalize([.2, .5, .8])]),
        metadata={"label": 2, "element_id": 1})
    metric.add_predictions(
        rm.common.types.ModelPredictions(predictions=[[.3, .7]]),
        metadata={"label": 0, "element_id": 2})
    metric_ls.add_predictions(
        rm.common.types.ModelPredictions(
            predictions=[_normalize([.3, .8, .7])]),
        metadata={"label": 0, "element_id": 2})
    expected_output = {
        _GCE_DEFAULT: {"gce": 0.51478150},
        _GCE_EXPLICIT_DEFAULT: {"gce": 0.5147815},
        _GCE_TEMP_SCALING_ALL: {"gce": 0.48694175, "beta": 0.4177706241607666},
        _GCE_ISOTONIC_REGR_ALL: {"gce": 0.0},
        _GCE_TEMP_SCALING_SPLIT: {"gce": 1.0, "beta": 0.4177706241607666},
        _GCE_ISOTONIC_REGR_SPLIT: {"gce": 1.0},
        "ece": {"ece": .45},
        "nll": {"nll": 0.5 * (-math.log(.8) - math.log(.3))},
        "brier": {"brier": 0.5 * (.2**2 + .2**2 + .7**2 + .7**2)},
        "temperature_scaling": {"beta": 0.4177706241607666},
    }[name]
    if name == _GCE_TEMP_SCALING_SPLIT:
      self.assertAlmostEqual(metric.result()["gce"],
                             expected_output["gce"])
      self.assertAllGreater(metric.result()["beta"], 0)
      self.assertAlmostEqual(metric_ls.result()["gce"],
                             expected_output["gce"])
      self.assertAllGreater(metric_ls.result()["beta"], 0)
    else:
      self.assertDictsAlmostEqual(metric.result(), expected_output)
      self.assertDictsAlmostEqual(metric_ls.result(), expected_output)

  @parameterized.parameters([(name,) for name in _UNCERTAINTY_METRICS])
  def test_tertiary_prediction(self, name):
    if "gce" == name[:3]:
      tempdir = self.create_tempdir().full_path
      # Need unmodified `name` to look up expected output.
      tempname = _with_tempdir(name, tempdir)
    else:
      tempname = name
    metric = rm.metrics.get(tempname, _get_info(3))
    metric_ls = rm.metrics.get(
        _with_labelset(tempname),
        rm.datasets.base.DatasetInfo(num_classes=4,
                                     appearing_classes=[1, 2, 3]))
    metric.add_predictions(
        rm.common.types.ModelPredictions(
            predictions=[[.2, .4, .4], [.5, .3, .2]]),
        metadata={"label": 2})
    metric_ls.add_predictions(
        rm.common.types.ModelPredictions(
            predictions=[_normalize([.5, .2, .4, .4]),
                         _normalize([.9, .5, .3, .2])]),
        metadata={"label": 3, "element_id": 1})
    metric.add_predictions(
        rm.common.types.ModelPredictions(predictions=[[.8, .15, .05]]),
        metadata={"label": 1, "element_id": 2})
    metric_ls.add_predictions(
        rm.common.types.ModelPredictions(
            predictions=[_normalize([.4, .8, .15, .05])]),
        metadata={"label": 2, "element_id": 2})
    expected_output = {
        _GCE_DEFAULT:
            {"gce": 0.6174544},
        _GCE_EXPLICIT_DEFAULT:
            {"gce": 0.6174544},
        _GCE_TEMP_SCALING_ALL:
            {"gce": 0.5588576, "beta": -0.23755066096782684},
        _GCE_ISOTONIC_REGR_ALL:
            {"gce": 0.23570},
        _GCE_TEMP_SCALING_SPLIT:
            {"gce": 1.0, "beta": -0.23755066096782684},
        _GCE_ISOTONIC_REGR_SPLIT:
            {"gce": 1.0},
        "ece":
            {"ece": 0.575},
        "nll":
            {"nll": -0.5 * (math.log((.4 + .2) / 2) + math.log(.15))},
        "brier":
            {"brier": 0.5 * (((.2 + .5) / 2)**2 + ((.4 + .3) / 2)**2 +
                             ((.6 + .8) / 2)**2 + .8**2 + .85**2 + .05**2)},
        "temperature_scaling": {"beta": -0.23755066096782684}
    }[name]
    if name == _GCE_TEMP_SCALING_SPLIT:
      self.assertAlmostEqual(metric.result()["gce"],
                             expected_output["gce"])
      self.assertAllLess(metric.result()["beta"], 0)
      self.assertAlmostEqual(metric_ls.result()["gce"],
                             expected_output["gce"])
      self.assertAllLess(metric_ls.result()["beta"], 0)
    else:
      self.assertDictsAlmostEqual(metric.result(), expected_output)
      self.assertDictsAlmostEqual(metric_ls.result(), expected_output)


class IsotonicRegressionTest(tf.test.TestCase):

  def test_IR_class(self):
    fit_predictions = np.array([[0.42610548, 0.41748077, 0.15641374],
                                [0.44766216, 0.47721294, 0.0751249],
                                [0.1862702, 0.15139402, 0.66233578],
                                [0.05753544, 0.8561222, 0.08634236],
                                [0.18697925, 0.29836466, 0.51465609]])
    fit_labels = np.array([0, 1, 2, 1, 2])
    scale_predictions = np.array([[0.1215652, 0.21415779, 0.66427702],
                                  [0.70361542, 0.21748313, 0.07890145],
                                  [0.46009217, 0.12798458, 0.41192324],
                                  [0.29240777, 0.31575023, 0.391842],
                                  [0.70334041, 0.13486871, 0.16179089]])
    scale_labels = np.array([2, 0, 0, 1, 0])
    pickle_path = self.create_tempdir()
    ir = rm.metrics.IsotonicRegression(pickle_path=pickle_path.full_path)
    ir.fit(fit_predictions, fit_labels)
    calibrated_predictions = ir.scale(scale_predictions)

    # Test calibration error should go down.
    ece_calibrated = rm.metrics.ExpectedCalibrationError()
    ece_calibrated.add_batch(calibrated_predictions, label=scale_labels)
    ece_scale = rm.metrics.ExpectedCalibrationError()
    ece_scale.add_batch(scale_predictions, label=scale_labels)
    self.assertLess(ece_calibrated.result()["ece"], ece_scale.result()["ece"])


class CRPSTest(tf.test.TestCase):

  def test_crps_increases_with_increasing_deviation_in_mean(self):
    """Assert that the CRPS score increases when we increase the mean.
    """
    tf.random.set_seed(1)

    nspacing = 10
    npredictive_samples = 10000
    ntrue_samples = 1000

    # (nspacing,npredictive_samples) samples from N(mu_i, 1)
    predictive_samples = tf.random.normal((nspacing, npredictive_samples))
    predictive_samples += tf.expand_dims(tf.linspace(0.0, 5.0, nspacing), 1)

    crps_samples = []
    for _ in range(ntrue_samples):
      labels = tf.random.normal((nspacing,))
      metric = rm.metrics.get("crps", _get_info(None))
      rm.metrics.add_batch(metric, predictive_samples, label=labels)
      crps_sample = metric.result()["crps"]
      crps_samples.append(crps_sample)

    crps_samples = tf.stack(crps_samples, 1)
    crps_average = tf.reduce_mean(crps_samples, axis=1)
    crps_average = crps_average.numpy()

    # The average should be monotonically increasing
    for i in range(1, len(crps_average)):
      crps_cur = crps_average[i]
      crps_prev = crps_average[i-1]
      self.assertLessEqual(crps_prev, crps_cur,
                           msg="CRPS violates monotonicity in mean")


class KerasECEMetricTest(parameterized.TestCase, tf.test.TestCase):

  _TEMPERATURES = [0.01, 1.0, 5.0]
  _NLABELS = [2, 4]
  _NSAMPLES = [8192, 16384]

  def _generate_perfect_calibration_logits(self, nsamples, nclasses):
    """Generate well distributed and well calibrated probabilities.

    Args:
      nsamples: int, >= 1, number of samples to generate.
      nclasses: int, >= 2, number of classes.

    Returns:
      logits: Tensor, shape (nsamples, nclasses), tf.float32, unnormalized log
        probabilities (logits) of the probabilistic predictions.
      labels: Tensor, shape (nsamples,), tf.int32, the true class labels.  Each
        element is in the range 0,..,nclasses-1.
    """
    tf.random.set_seed(1)

    logits = 2.0*tf.random.normal((nsamples, nclasses))
    py = tfp.distributions.Categorical(logits=logits)
    labels = py.sample()

    return logits, labels

  def _generate_random_calibration_logits(self, nsamples, nclasses):
    """Generate well distributed and poorly calibrated probabilities.

    Args:
      nsamples: int, >= 1, number of samples to generate.
      nclasses: int, >= 2, number of classes.

    Returns:
      logits: Tensor, shape (nsamples, nclasses), tf.float32, unnormalized log
        probabilities (logits) of the probabilistic predictions.
      labels: Tensor, shape (nsamples,), tf.int32, the true class labels.  Each
        element is in the range 0,..,nclasses-1.
    """
    tf.random.set_seed(1)

    logits = 2.0*tf.random.normal((nsamples, nclasses))
    py = tfp.distributions.Categorical(logits=logits)
    labels = py.sample()
    logits_other = 2.0*tf.random.normal((nsamples, nclasses))

    return logits_other, labels

  def test_binary_classification(self):
    num_bins = 10
    pred_probs = np.array([0.51, 0.45, 0.39, 0.66, 0.68, 0.29, 0.81, 0.85])
    # max_pred_probs: [0.51, 0.55, 0.61, 0.66, 0.68, 0.71, 0.81, 0.85]
    # pred_class: [1, 0, 0, 1, 1, 0, 1, 1]
    labels = np.array([0., 0., 0., 1., 0., 1., 1., 1.])
    n = len(pred_probs)

    # Bins for the max predicted probabilities are (0, 0.1), [0.1, 0.2), ...,
    # [0.9, 1) and are numbered starting at zero.
    bin_counts = np.array([0, 0, 0, 0, 0, 2, 3, 1, 2, 0])
    bin_correct_sums = np.array([0, 0, 0, 0, 0, 1, 2, 0, 2, 0])
    bin_prob_sums = np.array([0, 0, 0, 0, 0, 0.51 + 0.55, 0.61 + 0.66 + 0.68,
                              0.71, 0.81 + 0.85, 0])

    correct_ece = 0.
    bin_accs = np.array([0.] * num_bins)
    bin_confs = np.array([0.] * num_bins)
    for i in range(num_bins):
      if bin_counts[i] > 0:
        bin_accs[i] = bin_correct_sums[i] / bin_counts[i]
        bin_confs[i] = bin_prob_sums[i] / bin_counts[i]
        correct_ece += bin_counts[i] / n * abs(bin_accs[i] - bin_confs[i])

    metric = rm.metrics.uncertainty._KerasECEMetric(
        num_bins, name="ECE", dtype=tf.float64)
    self.assertLen(metric.variables, 3)

    ece1 = metric(labels, pred_probs)
    self.assertAllClose(ece1, correct_ece)

    actual_bin_counts = tf.convert_to_tensor(metric.counts)
    actual_bin_correct_sums = tf.convert_to_tensor(metric.correct_sums)
    actual_bin_prob_sums = tf.convert_to_tensor(metric.prob_sums)
    self.assertAllEqual(bin_counts, actual_bin_counts)
    self.assertAllEqual(bin_correct_sums, actual_bin_correct_sums)
    self.assertAllClose(bin_prob_sums, actual_bin_prob_sums)

    # Test various types of input shapes.
    metric.reset_states()
    metric.update_state(labels[:2], pred_probs[:2])
    metric.update_state(labels[2:6].reshape(2, 2),
                        pred_probs[2:6].reshape(2, 2))
    metric.update_state(labels[6:7], pred_probs[6:7])
    ece2 = metric(labels[7:, np.newaxis], pred_probs[7:, np.newaxis])
    ece3 = metric.result()
    self.assertAllClose(ece2, ece3)
    self.assertAllClose(ece3, correct_ece)

    actual_bin_counts = tf.convert_to_tensor(metric.counts)
    actual_bin_correct_sums = tf.convert_to_tensor(metric.correct_sums)
    actual_bin_prob_sums = tf.convert_to_tensor(metric.prob_sums)
    self.assertAllEqual(bin_counts, actual_bin_counts)
    self.assertAllEqual(bin_correct_sums, actual_bin_correct_sums)
    self.assertAllClose(bin_prob_sums, actual_bin_prob_sums)

  def test_binary_classification_binning_rule(self):
    num_bins = 10
    pred_probs = np.array([0.51, 0.45, 0.39, 0.66, 0.68, 0.29, 0.81, 0.85])
    # max_pred_probs: [0.51, 0.55, 0.61, 0.66, 0.68, 0.71, 0.81, 0.85]
    # pred_class: [1, 0, 0, 1, 1, 0, 1, 1]
    labels = np.array([0., 0., 0., 1., 0., 1., 1., 1.])
    n = len(pred_probs)

    custom_binning_score = np.array(
        [0.05, 0.11, 0.37, 0.52, 0.26, 0.47, 0.73, 0.23])
    # Bins for the max predicted probabilities are (0, 0.1), [0.1, 0.2), ...,
    # [0.9, 1) and are numbered starting at zero.
    bin_counts = np.array([1, 1, 2, 1, 1, 1, 0, 1, 0, 0])
    # pred_probs is correct at indices 1, 2, 3, 6, and 7.
    bin_correct_sums = np.array([0, 1, 1, 1, 0, 1, 0, 1, 0, 0])
    bin_prob_sums = np.array(
        [0.51, 1 - 0.45, 0.68 + 0.85, 1 - 0.39, 1 - 0.29, 0.66, 0, 0.81, 0, 0])

    correct_ece = 0.
    bin_accs = np.array([0.] * num_bins)
    bin_confs = np.array([0.] * num_bins)
    for i in range(num_bins):
      if bin_counts[i] > 0:
        bin_accs[i] = bin_correct_sums[i] / bin_counts[i]
        bin_confs[i] = bin_prob_sums[i] / bin_counts[i]
        correct_ece += bin_counts[i] / n * abs(bin_accs[i] - bin_confs[i])

    metric = rm.metrics.uncertainty._KerasECEMetric(
        num_bins, name="RebinnedECE", dtype=tf.float64)
    self.assertLen(metric.variables, 3)

    ece1 = metric(labels, pred_probs, custom_binning_score=custom_binning_score)
    self.assertAllClose(ece1, correct_ece)

    actual_bin_counts = tf.convert_to_tensor(metric.counts)
    actual_bin_correct_sums = tf.convert_to_tensor(metric.correct_sums)
    actual_bin_prob_sums = tf.convert_to_tensor(metric.prob_sums)
    self.assertAllEqual(bin_counts, actual_bin_counts)
    self.assertAllEqual(bin_correct_sums, actual_bin_correct_sums)
    self.assertAllClose(bin_prob_sums, actual_bin_prob_sums)

    # Test various types of input shapes.
    metric.reset_states()
    metric.update_state(
        labels[:1],
        pred_probs[:1],
        custom_binning_score=custom_binning_score[:1])
    metric.update_state(
        labels[1:5].reshape(2, 2),
        pred_probs[1:5].reshape(2, 2),
        custom_binning_score=custom_binning_score[1:5].reshape(2, 2))
    metric.update_state(
        labels[5:7],
        pred_probs[5:7],
        custom_binning_score=custom_binning_score[5:7])
    ece2 = metric(
        labels[7:, np.newaxis],
        pred_probs[7:, np.newaxis],
        custom_binning_score=custom_binning_score[7:, np.newaxis])
    ece3 = metric.result()
    self.assertAllClose(ece2, ece3)
    self.assertAllClose(ece3, correct_ece)

    actual_bin_counts = tf.convert_to_tensor(metric.counts)
    actual_bin_correct_sums = tf.convert_to_tensor(metric.correct_sums)
    actual_bin_prob_sums = tf.convert_to_tensor(metric.prob_sums)
    self.assertAllEqual(bin_counts, actual_bin_counts)
    self.assertAllEqual(bin_correct_sums, actual_bin_correct_sums)
    self.assertAllClose(bin_prob_sums, actual_bin_prob_sums)

  def test_binary_classification_keras_model(self):
    num_bins = 10
    pred_probs = np.array([0.51, 0.45, 0.39, 0.66, 0.68, 0.29, 0.81, 0.85])
    # max_pred_probs: [0.51, 0.55, 0.61, 0.66, 0.68, 0.71, 0.81, 0.85]
    # pred_class: [1, 0, 0, 1, 1, 0, 1, 1]
    labels = np.array([0., 0., 0., 1., 0., 1., 1., 1.])
    n = len(pred_probs)

    # Bins for the max predicted probabilities are (0, 0.1), [0.1, 0.2), ...,
    # [0.9, 1) and are numbered starting at zero.
    bin_counts = [0, 0, 0, 0, 0, 2, 3, 1, 2, 0]
    bin_correct_sums = [0, 0, 0, 0, 0, 1, 2, 0, 2, 0]
    bin_prob_sums = [0, 0, 0, 0, 0, 0.51 + 0.55, 0.61 + 0.66 + 0.68, 0.71,
                     0.81 + 0.85, 0]

    correct_ece = 0.
    bin_accs = [0.] * num_bins
    bin_confs = [0.] * num_bins
    for i in range(num_bins):
      if bin_counts[i] > 0:
        bin_accs[i] = bin_correct_sums[i] / bin_counts[i]
        bin_confs[i] = bin_prob_sums[i] / bin_counts[i]
        correct_ece += bin_counts[i] / n * abs(bin_accs[i] - bin_confs[i])

    metric = rm.metrics.uncertainty._KerasECEMetric(num_bins, name="ECE")
    self.assertLen(metric.variables, 3)

    model = tf.keras.models.Sequential([tf.keras.layers.Lambda(lambda x: 1*x)])
    model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=[metric])
    outputs = model.predict(pred_probs)
    self.assertAllClose(pred_probs.reshape([n, 1]), outputs)
    _, ece = model.evaluate(pred_probs, labels)
    self.assertAllClose(ece, correct_ece)

    actual_bin_counts = tf.convert_to_tensor(metric.counts)
    actual_bin_correct_sums = tf.convert_to_tensor(metric.correct_sums)
    actual_bin_prob_sums = tf.convert_to_tensor(metric.prob_sums)
    self.assertAllEqual(bin_counts, actual_bin_counts)
    self.assertAllEqual(bin_correct_sums, actual_bin_correct_sums)
    self.assertAllClose(bin_prob_sums, actual_bin_prob_sums)

  def test_ece_multiclass_classification(self):
    num_bins = 10
    pred_probs = [
        [0.31, 0.32, 0.27],
        [0.37, 0.33, 0.30],
        [0.30, 0.31, 0.39],
        [0.61, 0.38, 0.01],
        [0.10, 0.65, 0.25],
        [0.91, 0.05, 0.04],
    ]
    # max_pred_probs: [0.32, 0.37, 0.39, 0.61, 0.65, 0.91]
    # pred_class: [1, 0, 2, 0, 1, 0]
    labels = [1., 0, 0., 1., 0., 0.]
    n = len(pred_probs)

    # Bins for the max predicted probabilities are (0, 0.1), [0.1, 0.2), ...,
    # [0.9, 1) and are numbered starting at zero.
    bin_counts = [0, 0, 0, 3, 0, 0, 2, 0, 0, 1]
    bin_correct_sums = [0, 0, 0, 2, 0, 0, 0, 0, 0, 1]
    bin_prob_sums = [0, 0, 0, 0.32 + 0.37 + 0.39, 0, 0, 0.61 + 0.65, 0, 0, 0.91]

    correct_ece = 0.
    bin_accs = [0.] * num_bins
    bin_confs = [0.] * num_bins
    for i in range(num_bins):
      if bin_counts[i] > 0:
        bin_accs[i] = bin_correct_sums[i] / bin_counts[i]
        bin_confs[i] = bin_prob_sums[i] / bin_counts[i]
        correct_ece += bin_counts[i] / n * abs(bin_accs[i] - bin_confs[i])

    metric = rm.metrics.uncertainty._KerasECEMetric(
        num_bins, name="ECE", dtype=tf.float64)
    self.assertLen(metric.variables, 3)

    metric.update_state(labels[:4], pred_probs[:4])
    ece1 = metric(labels[4:], pred_probs[4:])
    ece2 = metric.result()
    self.assertAllClose(ece1, ece2)
    self.assertAllClose(ece2, correct_ece)

    actual_bin_counts = tf.convert_to_tensor(metric.counts)
    actual_bin_correct_sums = tf.convert_to_tensor(metric.correct_sums)
    actual_bin_prob_sums = tf.convert_to_tensor(metric.prob_sums)
    self.assertAllEqual(bin_counts, actual_bin_counts)
    self.assertAllEqual(bin_correct_sums, actual_bin_correct_sums)
    self.assertAllClose(bin_prob_sums, actual_bin_prob_sums)


class BrierDecompositionTest(parameterized.TestCase, tf.test.TestCase):
  _TEMPERATURES = [0.01, 1.0, 5.0]
  _NLABELS = [2, 4]
  _NSAMPLES = [8192, 16384]

  @parameterized.parameters(
      itertools.product(_TEMPERATURES, _NLABELS, _NSAMPLES)
  )
  def test_brier_decomposition(self, temperature, nlabels, nsamples):
    """Test the accuracy of the estimated Brier decomposition."""
    tf.random.set_seed(1)
    logits = tf.random.normal((nsamples, nlabels)) / temperature
    labels = tf.random.uniform((nsamples,), maxval=nlabels, dtype=tf.int32)

    metric = rm.metrics.get("brier_decomposition", _get_info(nlabels))
    rm.metrics.add_batch(metric,
                         tf.nn.softmax(logits, axis=-1).numpy(),
                         label=labels.numpy())
    result = metric.result()
    uncertainty = result["uncertainty"]
    resolution = result["resolution"]
    reliability = result["reliability"]

    # Recover an estimate of the Brier score from the decomposition
    brier = uncertainty - resolution + reliability

    # Estimate Brier score directly-
    metric = rm.metrics.get("brier", _get_info(nlabels))
    rm.metrics.add_batch(metric,
                         tf.nn.softmax(logits, axis=-1).numpy(),
                         label=labels.numpy())
    brier_direct = metric.result()["brier"]

    logging.info("Brier, n=%d k=%d T=%.2f, Unc %.4f - Res %.4f + Rel %.4f = "
                 "Brier %.4f,  Brier-direct %.4f",
                 nsamples, nlabels, temperature,
                 uncertainty, resolution, reliability,
                 brier, brier_direct)

    self.assertGreaterEqual(resolution, 0.0, msg="Brier resolution negative")
    self.assertGreaterEqual(reliability, 0.0, msg="Brier reliability negative")
    self.assertAlmostEqual(
        brier + 1, brier_direct, delta=1.0e-2,
        msg="Brier from decomposition (%.4f) and Brier direct (%.4f) disagree "
        "beyond estimation error." % (brier, brier_direct))


class SemiParametricCalibrationErrorTest(tf.test.TestCase):

  def test_zero_one(self):
    n = 2000
    probs = np.random.rand(n)
    calibration_error = 0.7 * probs ** 2 + 0.3 * probs
    # Simulate outcomes according to this model.
    labels = (np.random.rand(n) <= calibration_error).astype(np.float)
    metric = rm.metrics.get("semiparametric_ce(smoothing='spline')",
                            _get_info(None))
    rm.metrics.add_batch(metric, probs, label=labels)
    est = metric.result()["ce"]
    # est = ce.rms_calibration_error(probs, labels)
    self.assertGreaterEqual(est, 0)
    self.assertLessEqual(est, 1)

  def test_simple_call(self):
    n = 2000
    probs = np.random.rand(n)
    calibration_error = 0.7 * probs ** 2 + 0.3 * probs
    # Simulate outcomes according to this model.
    labels = (np.random.rand(n) <= calibration_error).astype(np.float)
    metric = rm.metrics.get("semiparametric_ce(smoothing='spline')",
                            _get_info(None))
    rm.metrics.add_batch(metric, probs, label=labels)
    est = metric.result()["ce"]
    self.assertGreaterEqual(est, 0)
    self.assertLessEqual(est, 1)

  def test_conf_int(self):
    n = 2000
    probs = np.random.rand(n)
    calibration_error = 0.7 * probs ** 2 + 0.3 * probs
    # Simulate outcomes according to this model.
    labels = (np.random.rand(n) <= calibration_error).astype(np.float)
    metric = rm.metrics.get("semiparametric_ce_ci(smoothing='spline')",
                            _get_info(None))
    rm.metrics.add_batch(metric, probs, label=labels)
    results = metric.result()
    self.assertGreaterEqual(results["low"], 0)
    self.assertLessEqual(results["low"], 1)
    self.assertGreaterEqual(results["high"], 0)
    self.assertLessEqual(results["high"], 1)

  def test_mean_plug_in(self):
    n = 2000
    probs = np.random.rand(n)
    calibration_error = 0.7 * probs ** 2 + 0.3 * probs
    # Continuous outcomes previously weren't allowed because StratifiedKFold
    # only allows discrete outcomes. Useful for testing to have an oracle
    # that passes in true calibration probabilities as outcomes, which are
    # continuous. Therefore, pass in a KFold object.
    metric = rm.metrics.SemiParametricCalibrationError(
        _get_info(None),
        smoothing="spline",
        fold_generator=sklearn.model_selection.KFold(5, shuffle=True))
    rm.metrics.add_batch(metric, probs, label=calibration_error)
    est = metric.result()["ce"]
    self.assertGreaterEqual(est, 0)
    self.assertLessEqual(est, 1)


def _get_adaptive_bins_test_parameters():
  np.random.seed(0)
  predictions = np.random.rand(500)
  # Test small number of bins:
  for num_bins in range(1, 50):
    yield {"predictions": predictions, "num_bins": num_bins}
  # Test large numbers of bins, including ones where some bins are empty:
  for num_bins in range(495, 505):
    yield {"predictions": predictions, "num_bins": num_bins}
  # Test where most bins are empty:
  yield {"predictions": np.random.rand(5), "num_bins": 30}


def _get_bin_counts(predictions, num_bins):
  bin_edges = rm.metrics.uncertainty._get_adaptive_bins(predictions, num_bins)
  # Bins should work with np.digitize:
  bin_indices = np.digitize(predictions, bin_edges)
  return np.bincount(bin_indices)


class GetAdaptiveBinsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(_get_adaptive_bins_test_parameters())
  def test_number_of_bins(self, predictions, num_bins):
    bin_counts = _get_bin_counts(predictions, num_bins)
    self.assertLen(bin_counts, num_bins)

  @parameterized.parameters(_get_adaptive_bins_test_parameters())
  def test_bins_include_all_datapoints(self, predictions, num_bins):
    bin_counts = _get_bin_counts(predictions, num_bins)
    self.assertLen(
        predictions, sum(bin_counts),
        msg="Sum of bin counts does not match length of predictions "
        f"({len(predictions)}): {bin_counts}")

  @parameterized.parameters(_get_adaptive_bins_test_parameters())
  def test_bins_have_similar_size(self, predictions, num_bins):
    bin_counts = _get_bin_counts(predictions, num_bins)
    self.assertAlmostEqual(
        np.max(bin_counts), np.min(bin_counts), delta=1,
        msg=f"Bin counts should differ by at most 1 but are {bin_counts}")


class GeneralCalibrationErrorTest(parameterized.TestCase, tf.test.TestCase):

  def test_consistency(self):
    probs = np.array([[0.42610548, 0.41748077, 0.15641374],
                      [0.44766216, 0.47721294, 0.0751249],
                      [0.1862702, 0.15139402, 0.66233578],
                      [0.05753544, 0.8561222, 0.08634236],
                      [0.18697925, 0.29836466, 0.51465609]])
    labels = np.array([0, 1, 2, 1, 2])
    metric = rm.metrics.GeneralCalibrationError(
        _get_info(3),
        num_bins=30, binning_scheme="even",
        class_conditional=False, max_prob=True, norm="l1", threshold=0.)
    rm.metrics.add_batch(metric, probs, label=labels)
    self.assertAlmostEqual(metric.result()["gce"], 0.412713502)

  def test_sweep(self):
    probs = np.array([[0.42610548, 0.41748077, 0.15641374],
                      [0.44766216, 0.47721294, 0.0751249],
                      [0.1862702, 0.15139402, 0.66233578],
                      [0.05753544, 0.8561222, 0.08634236],
                      [0.18697925, 0.29836466, 0.51465609]])
    labels = np.array([0, 1, 2, 1, 2])
    metric = rm.metrics.GeneralCalibrationError(
        _get_info(3), num_bins=None, binning_scheme="even",
        class_conditional=False, max_prob=True, norm="l1", threshold=0.)
    rm.metrics.add_batch(metric, probs, label=labels)
    self.assertAlmostEqual(metric.result()["gce"], 0.412713502)

  def test_binary_1d(self):
    probs = np.array([.91, .32, .66, .67, .57, .98, .41, .19])
    labels = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    metric = rm.metrics.GeneralCalibrationError(
        _get_info(2), num_bins=30, binning_scheme="even",
        class_conditional=False, max_prob=True, norm="l1", threshold=0.)
    rm.metrics.add_batch(metric, probs, label=labels)
    self.assertAlmostEqual(metric.result()["gce"], 0.18124999999999997)

  def test_binary_2d(self):
    probs = np.array(
        [.91, .32, .66, .67, .57, .98, .41, .19]).reshape(8, 1)
    labels = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    metric = rm.metrics.GeneralCalibrationError(
        _get_info(2), num_bins=30, binning_scheme="even",
        class_conditional=False, max_prob=True, norm="l1", threshold=0.)
    rm.metrics.add_batch(metric, probs, label=labels)
    self.assertAlmostEqual(metric.result()["gce"], 0.18124999999999997)

  def test_correctness_ece(self):
    num_bins = 10
    pred_probs = [
        [0.31, 0.32, 0.27],
        [0.37, 0.33, 0.30],
        [0.30, 0.31, 0.39],
        [0.61, 0.38, 0.01],
        [0.10, 0.65, 0.25],
        [0.91, 0.05, 0.04],
    ]
    # max_pred_probs: [0.32, 0.37, 0.39, 0.61, 0.65, 0.91]
    # pred_class: [1, 0, 2, 0, 1, 0]
    labels = [1., 0, 0., 1., 0., 0.]
    n = len(pred_probs)

    # Bins for the max predicted probabilities are (0, 0.1), [0.1, 0.2), ...,
    # [0.9, 1) and are numbered starting at zero.
    bin_counts = [0, 0, 0, 3, 0, 0, 2, 0, 0, 1]
    bin_correct_sums = [0, 0, 0, 2, 0, 0, 0, 0, 0, 1]
    bin_prob_sums = [0, 0, 0, 0.32 + 0.37 + 0.39, 0, 0, 0.61 + 0.65, 0, 0, 0.91]

    correct_ece = 0.
    bin_accs = [0.] * num_bins
    bin_confs = [0.] * num_bins
    for i in range(num_bins):
      if bin_counts[i] > 0:
        bin_accs[i] = bin_correct_sums[i] / bin_counts[i]
        bin_confs[i] = bin_prob_sums[i] / bin_counts[i]
        correct_ece += bin_counts[i] / n * abs(bin_accs[i] - bin_confs[i])

    metric_ece = rm.metrics.get("ece", _get_info(2))
    rm.metrics.add_batch(metric_ece, pred_probs, label=[int(i) for i in labels])
    self.assertAlmostEqual(correct_ece, metric_ece.result()["ece"])

  def test_correctness_rmsce(self):
    num_bins = 10
    pred_probs = [
        [0.31, 0.32, 0.27],
        [0.37, 0.33, 0.30],
        [0.30, 0.31, 0.39],
        [0.61, 0.38, 0.01],
        [0.10, 0.65, 0.25],
        [0.91, 0.05, 0.04],
    ]
    # max_pred_probs: [0.32, 0.37, 0.39, 0.61, 0.65, 0.91]
    # pred_class: [1, 0, 2, 0, 1, 0]
    labels = [1., 0, 0., 1., 0., 0.]
    n = len(pred_probs)

    # Adaptive bins, so every datapoint is on its own:
    bin_counts = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    bin_correct_sums = [1, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    bin_prob_sums = [0.32, 0, 0.37, 0, 0.39, 0.61, 0, 0.65, 0, 0.91]

    correct_ece = 0.
    bin_accs = [0.] * num_bins
    bin_confs = [0.] * num_bins
    for i in range(num_bins):
      if bin_counts[i] > 0:
        bin_accs[i] = bin_correct_sums[i] / bin_counts[i]
        bin_confs[i] = bin_prob_sums[i] / bin_counts[i]
        correct_ece += bin_counts[i] / n * np.square(bin_accs[i] - bin_confs[i])

    correct_rmsce = np.sqrt(correct_ece)

    metric_rmsce = rm.metrics.get("rmsce", _get_info(2))
    rm.metrics.add_batch(
        metric_rmsce, pred_probs, label=[int(i) for i in labels])

    self.assertAlmostEqual(correct_rmsce, metric_rmsce.result()["gce"])

  def generate_params():  # pylint: disable=no-method-argument
    # "self" object cannot be passes to parameterized.
    names = ["binning_scheme", "max_probs", "class_conditional",
             "threshold", "norm", "num_bins"]
    parameters = [["even", "adaptive"], [True, False], [True, False],
                  [0.0, 0.01], ["l1", "l2"], [30, None]]
    list(itertools.product(*parameters))
    count = 0
    dict_list = []
    for params in itertools.product(*parameters):
      param_dict = {}
      for i, v in enumerate(params):
        param_dict[names[i]] = v
      count += 1
      dict_list.append(param_dict)
    return dict_list

  @parameterized.parameters(generate_params())
  def test_generatable_metrics(self, class_conditional, threshold, max_probs,
                               norm, binning_scheme, num_bins):
    probs = np.array([[0.42610548, 0.41748077, 0.15641374, 0],
                      [0.44766216, 0.47721294, 0.0751249, 0],
                      [0.1862702, 0.15139402, 0.66233578, 0],
                      [0.05753544, 0.8561222, 0.08634236, 0],
                      [0.18697925, 0.29836466, 0.51465609, 0]])

    labels = np.array([0, 1, 2, 1, 2])
    metric = rm.metrics.GeneralCalibrationError(
        _get_info(4),
        binning_scheme=binning_scheme, max_prob=max_probs,
        class_conditional=class_conditional, threshold=threshold, norm=norm,
        num_bins=num_bins)
    rm.metrics.add_batch(metric, probs, label=labels)
    calibration_error = metric.result()["gce"]
    self.assertGreaterEqual(calibration_error, 0)
    self.assertLessEqual(calibration_error, 1)

  def test_get_bin_edges(self):
    bin_edges = rm.metrics.uncertainty._get_bin_edges([0, 0, 1, 1, 2, 2],
                                                      [.2, .4, .6, .7, .9, .95])
    self.assertAlmostEqual(bin_edges, [.5, .8, .95])

  def test_monotonic(self):
    probs = np.array([[0.42610548, 0.41748077, 0.15641374],
                      [0.44766216, 0.47721294, 0.0751249],
                      [0.1862702, 0.15139402, 0.66233578],
                      [0.05753544, 0.8561222, 0.08634236],
                      [0.18697925, 0.29836466, 0.51465609]])
    labels = np.array([0, 1, 2, 1, 2])

    bin_assignment = [0, 0, 1, 2, 1]
    is_monotonic = rm.metrics.uncertainty._is_monotonic(
        3, bin_assignment, labels)
    self.assertEqual(is_monotonic, False)

    bin_assign = rm.metrics.uncertainty._em_monotonic_sweep(
        probs.max(axis=1), labels)
    self.assertListEqual(bin_assign.tolist(), [0, 0, 1, 1, 0])

    bin_assign = rm.metrics.uncertainty._ew_monotonic_sweep(
        probs.max(axis=1), labels)
    self.assertListEqual(bin_assign.tolist(), [0, 0, 1, 1, 1])


class OracleCollaborativeAccuracyTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()

    self.num_bins = 10
    self.fraction = 0.4
    self.dtype = "float32"

    self.pred_probs = np.array([0.51, 0.45, 0.39, 0.66, 0.68, 0.29, 0.81, 0.85],
                               dtype=self.dtype)
    # max_pred_probs: [0.51, 0.55, 0.61, 0.66, 0.68, 0.71, 0.81, 0.85]
    # pred_class: [1, 0, 0, 1, 1, 0, 1, 1]
    self.labels = np.array([0., 0., 0., 1., 0., 1., 1., 1.], dtype=self.dtype)

  def test_oracle_collaborative_accuracy(self):
    # Bins for the max predicted probabilities are (0, 0.1), [0.1, 0.2), ...,
    # [0.9, 1) and are numbered starting at zero.
    bin_counts = np.array([0, 0, 0, 0, 0, 2, 3, 1, 2, 0])
    bin_correct_sums = np.array([0, 0, 0, 0, 0, 1, 2, 0, 2, 0])
    bin_prob_sums = np.array(
        [0, 0, 0, 0, 0, 0.51 + 0.55, 0.61 + 0.66 + 0.68, 0.71, 0.81 + 0.85, 0])
    # `(3 - 1)` refers to the rest examples in this bin
    # (minus the examples sent to the moderators), while `2/3` is
    # the accuracy in this bin.
    bin_collab_correct_sums = np.array(
        [0, 0, 0, 0, 0, 2, 1 * 1.0 + (3 - 1) * (2 / 3), 0, 2, 0])

    correct_acc = np.sum(bin_collab_correct_sums) / np.sum(bin_counts)

    metric = rm.metrics.uncertainty._KerasOracleCollaborativeAccuracyMetric(
        self.fraction, self.num_bins, name="collab_acc", dtype=tf.float64)

    acc = metric(self.labels, self.pred_probs)

    actual_bin_counts = tf.convert_to_tensor(metric.counts)
    actual_bin_correct_sums = tf.convert_to_tensor(metric.correct_sums)
    actual_bin_prob_sums = tf.convert_to_tensor(metric.prob_sums)
    actual_bin_bin_collab_correct_sums = tf.convert_to_tensor(
        metric.collab_correct_sums)

    self.assertAllEqual(bin_counts, actual_bin_counts)
    self.assertAllEqual(bin_correct_sums, actual_bin_correct_sums)
    self.assertAllClose(bin_prob_sums, actual_bin_prob_sums)
    self.assertAllClose(bin_collab_correct_sums,
                        actual_bin_bin_collab_correct_sums)

    self.assertAllClose(acc, correct_acc)

  def test_wrapped_oracle_collaborative_accuracy(self):
    # Bins for the max predicted probabilities are (0, 0.1), [0.1, 0.2), ...,
    # [0.9, 1) and are numbered starting at zero.
    bin_counts = np.array([0, 0, 0, 0, 0, 2, 3, 1, 2, 0])
    # `(3 - 1)` refers to the rest examples in this bin
    # (minus the examples sent to the moderators), while `2/3` is
    # the accuracy in this bin.
    bin_collab_correct_sums = np.array(
        [0, 0, 0, 0, 0, 2, 1 * 1.0 + (3 - 1) * (2 / 3), 0, 2, 0])

    correct_acc = np.sum(bin_collab_correct_sums) / np.sum(bin_counts)

    wrapped_metric = rm.metrics.uncertainty.OracleCollaborativeAccuracy(
        fraction=self.fraction, num_bins=self.num_bins)

    wrapped_metric.add_batch(self.pred_probs, label=self.labels)
    wrapped_metric_acc = wrapped_metric.result()["oracle_collaborative"]

    self.assertAllClose(wrapped_metric_acc, correct_acc)

  def test_wrapped_oracle_collaborative_accuracy_custom_binning_score(self):
    binning_score = tf.abs(self.pred_probs - 0.5)

    bin_counts = np.array([2, 3, 1, 2, 0, 0, 0, 0, 0, 0], dtype=self.dtype)
    bin_correct_sums = np.array([1, 2, 0, 2, 0, 0, 0, 0, 0, 0],
                                dtype=self.dtype)
    bin_prob_sums = np.array(
        [0.51 + 0.55, 0.61 + 0.66 + 0.68, 0.71, 0.81 + 0.85, 0, 0, 0, 0, 0, 0],
        dtype=self.dtype)
    # `(3 - 1)` refers to the rest examples in this bin
    # (minus the examples sent to the moderators), while `2/3` is
    # the accuracy in this bin.
    bin_collab_correct_sums = np.array(
        [2, 1 * 1.0 + (3 - 1) * (2 / 3), 0, 2, 0, 0, 0, 0, 0, 0],
        dtype=self.dtype)

    correct_acc = np.sum(bin_collab_correct_sums) / np.sum(bin_counts)

    metric = rm.metrics.uncertainty.OracleCollaborativeAccuracy(
        fraction=self.fraction, num_bins=self.num_bins)

    metric.add_batch(
        self.pred_probs, label=self.labels, custom_binning_score=binning_score)
    acc = metric.result()["oracle_collaborative"]

    actual_bin_counts = tf.convert_to_tensor(metric._metric.counts)
    actual_bin_correct_sums = tf.convert_to_tensor(metric._metric.correct_sums)
    actual_bin_prob_sums = tf.convert_to_tensor(metric._metric.prob_sums)
    actual_bin_bin_collab_correct_sums = tf.convert_to_tensor(
        metric._metric.collab_correct_sums)

    self.assertAllEqual(bin_counts, actual_bin_counts)
    self.assertAllEqual(bin_correct_sums, actual_bin_correct_sums)
    self.assertAllClose(bin_prob_sums, actual_bin_prob_sums)
    self.assertAllClose(bin_collab_correct_sums,
                        actual_bin_bin_collab_correct_sums)

    self.assertAllClose(acc, correct_acc)


if __name__ == "__main__":
  tf.test.main()
