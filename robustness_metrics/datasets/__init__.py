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

"""Module-level convenience functions."""

from robustness_metrics.datasets import base
from robustness_metrics.datasets import ood_detection
from robustness_metrics.datasets import tfds


def get(dataset_spec) -> base.Dataset:
  """Fetches a dataset from the dataset registry."""
  return base.registry.get_instance(dataset_spec)


def get_available_datasets():
  """Fetches dataset constructor from the dataset registry."""
  return base.registry.get_registered_subclasses()
