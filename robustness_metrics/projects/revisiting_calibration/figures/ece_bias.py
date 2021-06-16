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
"""Figures for "Revisiting Calibration of Modern Neural Networks".

This module contains figures showing how the accuracy-dependent bias of ECE
changes with number of bins used for estimating ECE.
"""
import re
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from robustness_metrics.projects.revisiting_calibration import display
from robustness_metrics.projects.revisiting_calibration import plotting
import seaborn as sns


def plot(df_main: pd.DataFrame,
         gce_prefix: str = plotting.STD_GCE_PREFIX,
         plot_confidence: bool = False,
         ) -> sns.axisgrid.FacetGrid:
  """Shows that bias can depend on accuracy."""
  # Select data:
  mask = df_main.Metric.str.startswith(gce_prefix.split("num_bins")[0])
  mask &= (df_main.ModelName.str.startswith("jft-r50-x1") |
           df_main.ModelName.str.startswith("jft-r101-x3"))
  mask &= df_main.DatasetName == "imagenet(split='validation[20%:]')"
  mask &= df_main.rescaling_method.isin(["none", "temperature_scaling"])
  mask &= df_main.RawModelName.str.contains("size")
  mask &= df_main.RawModelName.str.contains("steps")
  df_plot = df_main[mask].copy()

  df_plot["size"] = df_plot.RawModelName.map(
      lambda name: int(re.search(r"size=(\d+)", name).groups()[0]))
  df_plot["steps"] = df_plot.RawModelName.map(
      lambda name: int(re.search(r"steps=(\d+)", name).groups()[0]))
  df_plot["num_bins"] = df_plot.Metric.map(
      lambda metric: int(re.search(r"(?<=num_bins=)(\d+)", metric).groups()[0]))
  df_plot = plotting.add_optimal_temperature_as_rescaling_method(df_plot)

  # Remove outlier runs:
  df_plot = df_plot[df_plot.steps != 457032]

  color = sns.color_palette("colorblind", n_colors=1)[0]

  def subplot_fn_all_models_by_size(data, x, y, **kwargs):
    del kwargs
    ax = plt.gca()
    # Plot data:
    ax.scatter(
        data[x], data[y], s=3, c=color, alpha=0.75, zorder=30, label="data",
        linewidth=0)

  col_order = [15, 100, 5000]
  if plot_confidence:
    row_order = ["none", "temperature_scaling", "tau"]
  else:
    row_order = ["temperature_scaling"]

  g = plotting.FacetGrid(
      data=df_plot,
      sharex=False,
      sharey=False,
      dropna=False,
      col="num_bins",
      col_order=col_order,
      row="rescaling_method",
      row_order=row_order,
      height=1.2,
      aspect=0.8,
      margin_titles=True)

  g.map_dataframe(
      subplot_fn_all_models_by_size, x="imagenet_error", y="MetricValue")
  g.set_titles(col_template="{col_name:1.0f} bins", row_template="")

  # Format axes:
  for ax in g.axes.flatten():
    if ax.is_first_row():
      num_bins = col_order[plotting.col_num(ax)]
      ax.set_title(f"{num_bins} bins \n({40000/num_bins:1.0f} points/bin)",
                   fontsize=mpl.rcParams["axes.labelsize"])

    if not ax.is_last_row():
      ax.set_xticklabels([])

    if plotting.row_num(ax) < 2:
      # Set yaxis range to be the same:
      y = ax.collections[0].get_offsets()[:, 1]
      midpoint = np.mean([y.min(), y.max()])
      ax.set_ylim(midpoint-0.065/2, midpoint+0.065/2)
      ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.02))

    if row_order[plotting.row_num(ax)] == "none":
      # ax.set_ylim(0, 0.075)
      if ax.is_first_col():
        ax.set_ylabel(display.YLABEL_ECE_UNSCALED)

    elif row_order[plotting.row_num(ax)] == "temperature_scaling":
      if ax.is_first_col():
        ax.set_ylabel(display.YLABEL_ECE_TEMP_SCALED_SHORT)

    else:  # This is the over/underconfidence row.
      ax.set_ylim(0.85, 1.05)
      ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
      plotting.annotate_confidence_plot(ax)
      if ax.is_first_col():
        ax.set_ylabel("Optimal\ntemp. factor")

    ax.set_xlim(0.15, 0.35)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    plotting.show_spines(ax)
    if ax.is_last_row():
      ax.set_xlabel("Classification\nerror")
    ax.grid(axis="both")

  g.fig.tight_layout()
  return g
