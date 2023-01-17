# coding=utf-8
# Copyright 2023 The Robustness Metrics Authors.
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

"""An example numpy model."""
import numpy as np
from scipy.special import softmax


class Model:

  def __init__(self):
    self.weights = np.random.randn(3, 1000)

  def __call__(self, features):
    images = features["image"].numpy().reshape((-1, 224, 224, 3))
    means = np.mean(images, axis=(1, 2))
    logits = np.matmul(means, self.weights)
    return softmax(logits, axis=-1)


def create():
  return Model(), None
