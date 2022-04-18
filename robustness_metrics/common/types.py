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

"""The basic types and classes that we use in the project."""

from typing import Optional, List, Text, Any, Dict
import dataclasses


Features = Dict[Text, Any]


@dataclasses.dataclass(frozen=True)
class ModelPredictions:
  """Holds the predictions of a model on a specific dataset example.

  Properties:
    predictions: A list of predictions made on this example, each represented as
      a list of floats.
    time_in_s: The time in seconds the model took to make the predictions.
  """
  predictions: List[List[float]]
  time_in_s: Optional[float] = None
