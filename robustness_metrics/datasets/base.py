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

"""Robustness metrics.

Example usage:
```
from robustness_metrics import datasets
dataset = datasets.get("imagenet_a")
tf_dataset = dataset.load(preprocess_fn)
```
"""
import abc
import dataclasses
from typing import Callable, List, Optional

from robustness_metrics.common import registry
from robustness_metrics.common import types
import tensorflow as tf


@dataclasses.dataclass(frozen=True)
class DatasetInfo:
  num_classes: Optional[int]
  appearing_classes: Optional[List[int]] = None


class Dataset(metaclass=abc.ABCMeta):
  """The abstract class representing a dataset."""

  @abc.abstractproperty
  def info(self) -> DatasetInfo:
    """The properties of the dataset."""

  @abc.abstractmethod
  def load(self,
           preprocess_fn: Optional[Callable[[types.Features], types.Features]]
           ) -> tf.data.Dataset:
    """Loads the dataset.

    Note: The provided `preprocess_fn` gets *always* run in graph-mode.

    Args:
      preprocess_fn: The function used to preprocess the dataset before
        batching. Set to `None` for the per-dataset default.

    Returns:
      The pre-processed and batched dataset. Each element passed to
      `preprocess_fn` is a dictionary with the following fields:
         * "element_id": A unique integer assinged to each example.
         * "image": A (H, W, C)-tensor of type tf.uint8 holding the image.
         * "label": An int64 with the label.
         * "metadata": Additional data about an instance.
    """

registry = registry.Registry(Dataset)
