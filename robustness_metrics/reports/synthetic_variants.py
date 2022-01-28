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

# Lint as: python3
"""Report that evaluates accuracy on synthetic data."""

from robustness_metrics.reports import base


@base.registry.register("synthetic_variants")
class SyntheticVariantsReport(base.UnionReport):
  """Aggregated statistics over the Synthetic data.

  Synthetic data as introduced in
    On Robustness and Transferability of Convolutional Neural Networks
    Djolonga, Yung, Tschannen et al, 2020 (https://arxiv.org/pdf/2007.08558.pdf)

  This report contains the following Synthetic variants: `rotation`, `size`, and
  `location`. For each dataset, we compute the accuracy. Higher accuracy
  indicates better robustness.

  This report will contain the following keys, each indicating an accuracy:
  ['location(x1, y1)', 'location(x2, y2)', ..., 'location_average',
   'rotation(r1)', 'rotation(r2)', ..., 'rotation_average'
   'size(s1)', 'size(s2)', ..., 'size_average'] where x1, y1 are
   x- and y-coordinates, r1 is a rotation angle, and s1 is an object size as a
   percentage of the total image size.'
  """

  @property
  def required_measurements(self):
    for variant in ["rotation", "size", "location"]:
      dataset = f"synthetic(variant={variant!r})"
      yield base.MeasurementSpec(dataset, "synthetic")
