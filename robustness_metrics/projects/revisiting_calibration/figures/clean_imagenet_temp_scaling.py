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

"""Figures for "Revisiting Calibration of Modern Neural Networks".

This module contains figures comparing ECE on in-distribution data between
unscaled and temperature-scaled predictions.
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
  family_order = display.get_model_families_sorted()
  if add_guo:
    family_order.append("guo")

  # Set up figure:
  fig = plt.figure(figsize=(display.FULL_WIDTH/2, 1.4))
  spec = fig.add_gridspec(ncols=2, nrows=1)

  for ax_i, rescaling_method in enumerate(rescaling_methods):

    df_plot, cmap = _get_data(df_main, gce_prefix, family_order,
                              rescaling_methods=[rescaling_method])
    ax = fig.add_subplot(spec[:, ax_i])
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
          alpha=0.7,
          linewidth=0.0,
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
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.01))
    ax.set_ylim(bottom=0.01, top=0.09)
    ax.set_xlim(0.05, 0.5)
    ax.set_xlabel(display.XLABEL_INET_ERROR)
    if rescaling_method == "none":
      ax.set_title("Unscaled")
    elif rescaling_method == "temperature_scaling":
      ax.set_title("Temperature-scaled")
      ax.set_yticklabels("")
    if plotting.is_first_col(ax):
      ax.set_ylabel("ECE")

  # Model family legend:
  handles, labels = plotting.get_model_family_legend(big_ax, family_order)
  legend = big_ax.legend(
      handles=handles, labels=labels, loc="upper right", frameon=True,
      labelspacing=0.3, handletextpad=0.1, borderpad=0.3, fontsize=4)
  legend.get_frame().set_linewidth(mpl.rcParams["axes.linewidth"])
  legend.get_frame().set_edgecolor("lightgray")

  plotting.apply_to_fig_text(fig, display.prettify)
  plotting.apply_to_fig_text(fig, lambda x: x.replace("EfficientNet", "EffNet"))

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
