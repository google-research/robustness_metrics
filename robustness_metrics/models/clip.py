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

"""A zero-shot CLIP classification model.

For more information on the models please refer to

  https://openai.com/blog/clip/

and the PyTorch documentation: https://github.com/openai/CLIP/

You need `CLIP` to be on sys.path before loading this module. Note that the
model will be downloaded on the first run.

We are using a single prompt for each query: "This is a photo of a {label}".
The labels can be found in `imagenet.json`, in the same directory as this file.
"""
import json
import os

import clip
import numpy as np
from robustness_metrics.common import pipeline_builder
import torch


def create(network):
  """Loads the CLIP model."""
  json_path = os.path.join(os.path.dirname(__file__), "imagenet.json")
  with open(json_path, "r") as fp:
    imagenet_labels = json.load(fp)

  with torch.set_grad_enabled(False):
    model, _ = clip.load(network, device="cuda", jit=False)
    model = model.eval()

  prompts = clip.tokenize([
      f"This is a photo of a {label}" for label in imagenet_labels])

  with torch.no_grad():
    prompts_features = model.encode_text(prompts.cuda()).float()
    prompts_features /= prompts_features.norm(dim=-1, keepdim=True)

  image_mean = [0.48145466, 0.4578275, 0.40821073]
  image_std = [0.26862954, 0.26130258, 0.27577711]

  def call(features):
    # Normalize according to the documentation. Note that the pre-processing
    # will already have the range normalized to [0, 1].
    images_normalized = (features["image"] - image_mean) / image_std
    # Reshape from [batch, h, w, c] -> [batch, c, h, w]
    images_torch = torch.tensor(
        np.transpose(images_normalized, [0, 3, 1, 2]).astype(np.float32))
    with torch.no_grad():
      image_features = model.encode_image(images_torch.to("cuda")).float()
      image_features /= image_features.norm(dim=-1, keepdim=True)
      similarities = image_features @ prompts_features.T
      # The 100 (inv temperature) comes from the released code.
      return (100.0 * similarities).softmax(dim=-1).cpu().numpy()

  input_resolution = model.visual.input_resolution
  preprocess_config = (f"resize_small({input_resolution})|"
                       f"central_crop({input_resolution})|"
                       f"value_range(0,1)")
  preprocess_fn = pipeline_builder.get_preprocess_fn(
      preprocess_config, remove_tpu_dtypes=False)
  return call, preprocess_fn
