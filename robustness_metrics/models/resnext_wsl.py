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
"""Weakly Supervised ResNext.

For more information on the models please refer to the paper

  Mahajan, Dhruv, et al.
  "Exploring the limits of weakly supervised pretraining."
  Proceedings of the European Conference on Computer Vision (ECCV). 2018.

and the PyTorch documentation:

   https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/

Note that the model will be downloaded on the first run.
"""
from absl import logging
import numpy as np
from robustness_metrics.common import pipeline_builder
import torch


def create(variant):
  """Loads the model.

  Args:
    variant: One of 32x8d,32x16d,32x32d,32x48d.
  Returns:
    The model and the pre-processing function.
  """
  with torch.set_grad_enabled(False):
    model = torch.hub.load(
        "facebookresearch/WSL-Images", f"resnext101_{variant}_wsl").eval()

  with_cuda = torch.cuda.is_available()
  if with_cuda:
    model.to("cuda")
  else:
    logging.warn("Running on CPU, no CUDA detected.")

  def call(features):
    images = features["image"].numpy()
    # Normalize according to the documentation. Note that the pro-processing
    # will already have the range normalized to [0, 1].
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    images_normalized = (images - mean) / std
    # Reshape from [batch, h, w, c] -> [batch, c, h, w]
    images_normalized_bchw = np.transpose(
        images_normalized, [0, 3, 1, 2]).astype(np.float32).copy()
    with torch.no_grad():
      images_torch = torch.from_numpy(images_normalized_bchw)
      if with_cuda:
        images_torch = images_torch.to("cuda")
      logits = model(images_torch)
      return torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

  preprocess_config = "resize_small(256)|central_crop(224)|value_range(0,1)"
  preprocess_fn = pipeline_builder.get_preprocess_fn(
      preprocess_config, remove_tpu_dtypes=True)
  return call, preprocess_fn
