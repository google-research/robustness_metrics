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
"""Robustness metrics.

This module provides a set of metrics, which accept `ModelPrediction`s and
compute a dictionary of floating point numbers. Each metric is registered under
a unique identifier, and the method `get` can be used to obtain the Metric
subclass given the identifier. Example usage:
```
metric = metrics.get("accuracy")()
metric.add_predictions(
  ModelPredictions(element_id=0,
                   metadata={"label": 1},
                   predictions=[[.2, .8], [.7, .3]]))
results = model.results()
print(f"Results: {results!r}")
```
"""
from typing import Text, Type

from robustness_metrics.metrics import base
from robustness_metrics.metrics import timing
from robustness_metrics.metrics import uncertainty


def get(metric_name: Text, dataset_info):
  """Returns the subclass of `Metric` that has the given name.

  Args:
    metric_name: The name of the metric.
    dataset_info: The dataset info to be passed to the metric initializer.
  Returns:
    The subclass if `Metric` that has the given name.
  Raises:
    KeyError: If there is no metric registered under the given name.
  """
  # Register your metric by adding an entry in the dictionary below.
  return base.registry.get_instance(metric_name, dataset_info=dataset_info)
