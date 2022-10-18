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

"""Figures for "Revisiting Calibration of Modern Neural Networks".

This module contains figures comparing ECE of BiT-models pretrained on different
datasets.
"""

from typing import List, Optional, Tuple

import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
from robustness_metrics.projects.revisiting_calibration import display
from robustness_metrics.projects.revisiting_calibration import plotting
from robustness_metrics.projects.revisiting_calibration import utils


def plot(df_main: pd.DataFrame,
         gce_prefix: str = plotting.STD_GCE_PREFIX,
         rescaling_method: str = "temperature_scaling",
         add_guo: bool = False) -> mpl.figure.Figure:
  """Plots acc/calib and reliability diagrams on clean ImageNet (Figure 1)."""
  rescaling_methods = ["none", rescaling_method]
  family_order = display.get_model_families_sorted(
      ["mixer", "vit", "bit", "simclr"])
  if add_guo:
    family_order.append("guo")

  # Set up figure:
  fig = plt.figure(figsize=(display.FULL_WIDTH/2, 2))
  spec = fig.add_gridspec(ncols=3, nrows=2)

  for col, bit_version in enumerate(
      ["BiT-ImageNet", "BiT-ImageNet21k", "BiT-JFT"]):
    # pylint: disable=g-long-lambda
    if bit_version == "BiT-ImageNet":
      display.get_standard_model_list = lambda: [
          m for m in display.MODEL_SIZE.keys()
          if not (m.startswith("bit-imagenet21k-") or m.startswith("bit-jft-"))
      ]
    elif bit_version == "BiT-ImageNet21k":
      display.get_standard_model_list = lambda: [
          m for m in display.MODEL_SIZE.keys()
          if not (m.startswith("bit-imagenet-") or m.startswith("bit-jft-"))
      ]
    elif bit_version == "BiT-JFT":
      display.get_standard_model_list = lambda: [
          m for m in display.MODEL_SIZE.keys() if not (m.startswith(
              "bit-imagenet-") or m.startswith("bit-imagenet21k-"))
      ]
    else:
      raise ValueError(f"Unknown BiT version: {bit_version}")
    # pylint: enable=g-long-lambda

    for row, rescaling_method in enumerate(rescaling_methods):

      df_plot, cmap = _get_data(df_main, gce_prefix, family_order,
                                rescaling_methods=[rescaling_method])
      ax = fig.add_subplot(spec[row, col])
      big_ax = ax
      for i, family in enumerate(family_order):
        if family == "guo":
          continue
        data_sub = df_plot[df_plot.ModelFamily == family]
        if data_sub.empty:
          continue
        ax.scatter(
            data_sub["downstream_error"],
            data_sub["MetricValue"],
            s=plotting.model_to_scatter_size(data_sub.model_size),
            c=data_sub.family_index,
            cmap=cmap,
            vmin=0,
            vmax=len(family_order),
            marker=utils.assert_and_get_constant(data_sub.family_marker),
            linewidth=0.5,
            alpha=1.0 if "bit" in family else 0.5,
            zorder=100 - i,  # Z-order is same as model family order.
            label=family)

      # Manually add Guo et al data:
      # From Table 1 and Table S2 in https://arxiv.org/pdf/1706.04599.pdf.
      # First model is DenseNet161, second is ResNet152.
      if add_guo:
        size = plotting.model_to_scatter_size(1)
        color = [len(family_order) - 1] * 2
        marker = "x"
        if rescaling_method == "none":
          ax.scatter([0.2257, 0.2231], [0.0628, 0.0548],
                     s=size, c=color, marker=marker, alpha=0.7, label="guo")
        if rescaling_method == "temperature_scaling":
          ax.scatter([0.2257, 0.2231], [0.0199, 0.0186],
                     s=size, c=color, marker=marker, alpha=0.7, label="guo")

      plotting.show_spines(ax)
      # Aspect ratios are tuned manually for display in the paper:
      ax.set_anchor("N")
      ax.grid(False, which="minor")
      ax.grid(True, axis="both")
      ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.01))
      ax.set_ylim(bottom=0.0, top=0.09)
      ax.set_xlim(0.05, 0.3)
      ax.set_xlabel(display.XLABEL_INET_ERROR)
      if plotting.is_first_row(ax):
        ax.set_title(bit_version, fontsize=6)
        ax.set_xlabel("")
        ax.set_xticklabels("")
      else:
        ax.set_ylim(bottom=0.0, top=0.05)
      if plotting.is_first_col(ax):
        if rescaling_method == "none":
          ax.set_ylabel(display.YLABEL_ECE_UNSCALED)
        elif rescaling_method == "temperature_scaling":
          ax.set_ylabel(display.YLABEL_ECE_TEMP_SCALED)
      else:
        ax.set_yticklabels("")

  fig.tight_layout(pad=0.5)

  # Model family legend:
  handles, labels = plotting.get_model_family_legend(big_ax, family_order)

  plotting.apply_to_fig_text(fig, display.prettify)
  plotting.apply_to_fig_text(fig, lambda x: x.replace("EfficientNet", "EffNet"))

  legend = fig.axes[0].legend(
      handles=handles,
      labels=labels,
      loc="upper center",
      title="Model family",
      bbox_to_anchor=(0.55, -0.025),
      frameon=True,
      bbox_transform=fig.transFigure,
      ncol=len(family_order),
      handletextpad=0.1)
  legend.get_frame().set_linewidth(mpl.rcParams["axes.linewidth"])
  legend.get_frame().set_edgecolor("lightgray")
  plotting.apply_to_fig_text(fig, display.prettify)

  return fig


def _get_data(
    df_main: pd.DataFrame,
    gce_prefix: str,
    family_order: List[str],
    rescaling_methods: Optional[List[str]] = None,
    dataset_name: str = "imagenet(split='validation[20%:]')"
) -> Tuple[pd.DataFrame, mpl.colors.ListedColormap]:
  """Selects data for plotting."""
  # Select data:
  mask = df_main.Metric.str.startswith(gce_prefix)
  mask &= df_main.ModelName.isin(display.get_standard_model_list())
  mask &= df_main.DatasetName.isin([dataset_name])
  mask &= df_main.rescaling_method.isin(
      rescaling_methods or ["temperature_scaling"])
  mask &= df_main.ModelFamily.isin(family_order)
  df_plot = df_main[mask].copy()
  df_plot, cmap = display.add_display_data(df_plot, family_order)
  return df_plot, cmap
