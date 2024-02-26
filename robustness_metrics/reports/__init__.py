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

"""Reports to compute robustness scores of machine learning models."""
from typing import Text, Type
from robustness_metrics.reports import base
from robustness_metrics.reports import cifar_variants
from robustness_metrics.reports import imagenet_variants
from robustness_metrics.reports import ood_detection
from robustness_metrics.reports import plex
from robustness_metrics.reports import synthetic_variants


def get(report_spec) -> base.Report:
  """Load the report registered under the different name.

  Args:
    report_spec: A specification of the report to be constructed.

  Returns:
    A report constructed using the given spec.
  """
  return base.registry.get_instance(report_spec)
