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

"""Anti aliased CNNs.

For more information on the models please refer to

  https://github.com/adobe/antialiased-cnns

You need `antialiased-cnns` to be on sys.path before loading this module.
Note that the model will be downloaded on the first run.

The model scores 77.2% accuracy on ImageNet and 65.0% on ImageNet-V2.
"""
import antialiased_cnns
import numpy as np
from robustness_metrics.common import pipeline_builder
import torch


def create(config=None):
  """Loads the Anti Alias model."""

  del config  # Unused argument

  with torch.set_grad_enabled(False):
    model = antialiased_cnns.resnet50(pretrained=True)
    model = model.eval()

  image_mean = [0.485, 0.456, 0.406]
  image_std = [0.229, 0.224, 0.225]

  def call(features):
    # Normalize according to the documentation. Note that the pre-processing
    # will already have the range normalized to [0, 1].
    images_normalized = (features["image"] - image_mean) / image_std

    # Reshape from [batch, h, w, c] -> [batch, c, h, w]
    images_torch = torch.tensor(
        np.transpose(images_normalized, [0, 3, 1, 2]).astype(np.float32))

    with torch.no_grad():
      logits = model(images_torch)
      return logits.softmax(dim=-1).cpu().numpy()

  preprocess_config = ("resize_small(256)|"
                       "central_crop(224)|"
                       "value_range(0,1)")
  preprocess_fn = pipeline_builder.get_preprocess_fn(
      preprocess_config, remove_tpu_dtypes=False)
  return call, preprocess_fn
