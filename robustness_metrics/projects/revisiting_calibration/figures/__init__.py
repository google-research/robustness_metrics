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

"""Figures for 'Revisiting Calibration of Modern Neural Networks'.

Each figure is defined as a separate Python file in the "figures" directory. The
file should define a "plot" function which generates the main figure. It may
also define functions for additional plot variants and helper functions.

New figure files should be imported here and added to __all__ to allow for
easy importing and automatic testing.
"""

import pandas as _pd

from robustness_metrics.projects.revisiting_calibration.figures import acc_calib_regression
from robustness_metrics.projects.revisiting_calibration.figures import clean_imagenet_and_reliability_diags
from robustness_metrics.projects.revisiting_calibration.figures import clean_imagenet_temp_scaling
from robustness_metrics.projects.revisiting_calibration.figures import clean_imagenet_temp_scaling_bit_pretrain_comparison
from robustness_metrics.projects.revisiting_calibration.figures import ece_bias
from robustness_metrics.projects.revisiting_calibration.figures import imagenet_c
from robustness_metrics.projects.revisiting_calibration.figures import ood_comparison
from robustness_metrics.projects.revisiting_calibration.figures import pretrain_size_steps_sweep

# Raise error for potentially unsafe chained assignments in pandas:
_pd.options.mode.chained_assignment = "raise"


__all__ = [
    "acc_calib_regression",
    "clean_imagenet_and_reliability_diags",
    "clean_imagenet_temp_scaling",
    "clean_imagenet_temp_scaling_bit_pretrain_comparison",
    "ece_bias",
    "imagenet_c",
    "ood_comparison",
    "pretrain_size_steps_sweep",
]
