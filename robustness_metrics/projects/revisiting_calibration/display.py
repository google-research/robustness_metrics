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
"""Constants and code defining how to display 'Revisiting Calibration' data."""
from typing import List, Optional, Tuple

import matplotlib as mpl
import pandas as pd

FULL_WIDTH = 5.5  # Full figure width in the manuscript, in inches.
XLABEL_INET_ERROR = "ImageNet error"
XLABEL_INET_C_ERROR = "ImageNet-C error"
XLABEL_CLASSIFICATION_ERROR = "Classification error"
YLABEL_ECE_UNSCALED = "ECE\n(no recalibration)"
YLABEL_ECE_TEMP_SCALED = "ECE\n(temperature-scaled)"
YLABEL_ECE_TEMP_SCALED_SHORT = "ECE\n(temp.-scaled)"
YLABEL_ECE_BETA_SCALED = "ECE\n(beta-scaled)"
YLABEL_TEMP_FACTOR = "Optimal\ntemperature factor"
YLABEL_TEMP_FACTOR_SHORT = "Optimal\ntemp. factor"


MODEL_SIZE = {
    "vit-b/32": 1,
    "vit-b/16": 1.75,
    "vit-l/32": 2.5,
    "vit-l/16": 3.25,
    "vit-h/14": 4,

    "bit-jft-r50-x1": 1,
    "bit-jft-r101-x1": 1.75,
    "bit-jft-r50-x3": 2.5,
    "bit-jft-r101-x3": 3.25,
    "bit-jft-r152-x4-480": 4,

    "bit-imagenet-r50-x1": 1,
    "bit-imagenet-r101-x1": 1.75,
    "bit-imagenet-r50-x3": 2.5,
    "bit-imagenet-r101-x3": 3.25,
    "bit-imagenet-r152-x4-480": 4,

    "jax-bit-imagenet-r50-x1": 1.0,

    "bit-imagenet21k-r50-x1": 1,
    "bit-imagenet21k-r101-x1": 1.75,
    "bit-imagenet21k-r50-x3": 2.5,
    "bit-imagenet21k-r101-x3": 3.25,
    "bit-imagenet21k-r152-x4-480": 4,

    "simclr-1x-fine-tuned-100": 1,
    "simclr-2x-fine-tuned-100": 2,
    "simclr-4x-fine-tuned-100": 4,

    "efficientnet-noisy-student-b1": 1.0,
    "efficientnet-noisy-student-b3": 2.0,
    "efficientnet-noisy-student-b5": 3.0,
    "efficientnet-noisy-student-b7": 4.0,

    "efficientnet-b0": 1.0,
    "efficientnet-b4": 2.5,

    "alexnet": 1.0,
    "vgg": 1.0,

    "wsl_32x8d": 1,
    "wsl_32x16d": 2,
    "wsl_32x32d": 3,
    "wsl_32x48d": 4,

    "clip_vit_b32": 1.0,
    "clip_r50": 1.0,
    "clip_r101": 1.75,
    "clip_r50x4": 3.0,

    "mixer/jft-300m/B/16": 1.75,
    "mixer/jft-300m/L/16": 3.25,
    "mixer/jft-300m/H/14": 4,
    "mixer/jft-2.5b/H/14": 5,
}

_MODEL_ORDER = [
    "mixer",
    "vit",
    "bit",
    "wsl",
    "simclr",
    "clip",
    "alexnet",
]

_MODEL_MARKERS = {
    "vit": "o",
    "bit": "s",
    "mixer": "^",
    "simclr": "D",
    "efficientnet-noisy-student": "h",
    "alexnet": "X",
    "vgg": "P",
    "clip": "*",
    "wsl": "p",
    "guo": "x",
}

OOD_DATASET_ORDER = [
    "imagenet_v2(variant='MATCHED_FREQUENCY')",
    "imagenet_r",
    "imagenet_a",
]


def get_standard_model_list():
  """Returns the list of all models shown in the main paper figures."""
  return [m for m in MODEL_SIZE.keys() if not m.startswith("bit-imagenet")]


def get_model_families_sorted(model_list=None):
  """Returns a list of model family names in the canonical order."""
  if model_list is None:
    return list(_MODEL_ORDER)  # Return a copy.
  msg = f"Canonical order is incomplete: {model_list}"
  assert set(model_list).issubset(_MODEL_ORDER), msg
  return [m for m in _MODEL_ORDER if m in model_list]


def get_model_family_color(model_family):
  """Returns the canonical color for a model family."""
  # Derived from sns.color_palette("colorblind").
  canonical_colors = {
      "vit": "#0173B2",
      "bit": "#DE8F05",
      "simclr": "#029E73",
      "efficientnet-noisy-student": "#555555",
      "wsl": "#CC78BC",
      "clip": "#CA9161",
      "vgg": "#949494",
      "alexnet": "#949494",
      "mixer": "#D55E00",
      "guo": "#000000",
  }
  assert model_family in canonical_colors, f"Specify color for {model_family}."
  return canonical_colors[model_family]


def get_model_family_marker(model_family):
  assert model_family in _MODEL_MARKERS, f"Specify marker for {model_family}."
  return _MODEL_MARKERS[model_family]


def prettify(s):
  """Universal function to prettify all strings.

  Intended use:
  Never directly change strings in the dataframe because it's confusing.

  Instead, use `plotting.apply_to_fig_text(fig_handle, prettify_names)` on any
  figure to automaticaly change all labels, titles and other strings in the
  figure to the pretty version.

  Args:
    s: String/name as used in the database.

  Returns:
    Pretty string for use in a figure.
  """
  pretty_name = {
      # Datasets:
      "imagenet": "ImageNet",
      "imagenet(split='validation[20%:]')": "ImageNet",
      "imagenet_a": "ImageNet-A",
      "imagenet_r": "ImageNet-R",
      "imagenet_v2": "ImageNet-v2",
      "imagenet_v2(variant='MATCHED_FREQUENCY')": "ImageNet-v2",
      "objectnet": "ObjectNet",
      "jft": "JFT-300M",
      "imagenet21k": "ImageNet-21k",

      # Model families:
      "vit": "ViT",
      "bit": "BiT",
      "simclr": "SimCLR",
      "efficientnet-noisy-student": "EfficientNet-NS",
      "efficientnet-b0": "EfficientNet-B0",
      "efficientnet-b4": "EfficientNet-B4",
      "jax-bit-imagenet-r50-x1": "BiT-S R50x1",
      "alexnet": "AlexNet",
      "clip": "CLIP",
      "vgg": "VGG",
      "wsl": "ResNeXt WSL",
      "mixer": "MLP-Mixer",
      "guo": "Guo et al.",

      # Model names:
      "vit-b/32": "ViT-B/32",
      "vit-b/16": "ViT-B/16",
      "vit-l/32": "ViT-L/32",
      "vit-l/16": "ViT-L/16",
      "vit-h/14": "ViT-H/14",

      "bit-jft-r50-x1": "BiT-L (R50x1)",
      "bit-jft-r101-x1": "BiT-L (R101x1)",
      "bit-jft-r50-x3": "BiT-L (R50x3)",
      "bit-jft-r101-x3": "BiT-L (R101x3)",
      "bit-jft-r152-x4-480": "BiT-L (R152x4)",

      "simclr-1x-fine-tuned-100": "SimCLR-1x",
      "simclr-2x-fine-tuned-100": "SimCLR-2x",
      "simclr-4x-fine-tuned-100": "SimCLR-4x",

      "efficientnet-noisy-student-b1": "EffNet-B1-NS",
      "efficientnet-noisy-student-b3": "EffNet-B3-NS",
      "efficientnet-noisy-student-b5": "EffNet-B5-NS",
      "efficientnet-noisy-student-b7": "EffNet-B7-NS",

      "wsl_32x8d": "WSL 32x8d",
      "wsl_32x16d": "WSL 32x16d",
      "wsl_32x32d": "WSL 32x32d",
      "wsl_32x48d": "WSL 32x48d",

      "clip_r50": "CLIP R50",
      "clip_r101": "CLIP R101",
      "clip_r50x4": "CLIP R50x4",
      "clip_vit_b32": "CLIP ViT-B/32",

      "mixer/jft-300m/B/16": "MLP-Mixer-B/16",
      "mixer/jft-300m/L/16": "MLP-Mixer-L/16",
      "mixer/jft-300m/H/14": "MLP-Mixer-H/14",
      "mixer/jft-2.5b/H/14": "MLP-Mixer-H/14 (2.5B)",

      # Other:
      "size varies": "Pretraining dataset size\n(num. examples)",
      "steps varies": "Pretraining duration\n(steps)",
      "temperature_scaling": "Temperature scaling",
  }

  # If no pretty version is defined, return the original string:
  return pretty_name.get(s, s)


def add_display_data(
    df: pd.DataFrame,
    family_order: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, mpl.colors.ListedColormap]:
  """Add plotting-related data to the dataframe."""
  df = df.copy()
  df["model_size"] = df.ModelName.map(MODEL_SIZE)
  df["family_marker"] = df.ModelFamily.map(get_model_family_marker)
  df["family_color"] = df.ModelFamily.map(get_model_family_color)
  family_order = family_order or get_model_families_sorted(
      df.ModelFamily.unique())
  df["family_index"] = df.ModelFamily.map(family_order.index)
  cmap = mpl.colors.ListedColormap(
      [get_model_family_color(f) for f in family_order], N=len(family_order))
  return df, cmap
