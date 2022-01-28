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
"""Figures for 'Revisiting Calibration of Modern Neural Networks'.

This module contains figures showing how accuracy and calibration depend on the
amount of pretraining data and the number of pretraining steps.
"""
import re
from typing import Sequence

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from robustness_metrics.projects.revisiting_calibration import display
from robustness_metrics.projects.revisiting_calibration import plotting
from robustness_metrics.projects.revisiting_calibration import utils
import seaborn as sns


def plot(df_main: pd.DataFrame,
         upstream_datasets: Sequence[str] = ("imagenet21k", "jft"),
         gce_prefix: str = plotting.STD_GCE_PREFIX) -> sns.axisgrid.FacetGrid:
  """Plot acc. and calibration with changing pretraining data size and steps."""
  df_plot = _get_data(df_main, upstream_datasets, gce_prefix)
  df_plot = df_plot[df_plot.rescaling_method == "temperature_scaling"]

  sns.set_style("ticks")
  g = sns.FacetGrid(
      data=df_plot,
      row="upstream_dataset",
      row_order=upstream_datasets,
      col="varying_key",
      col_order=["size", "steps"],
      sharex=True,
      sharey=False,
      dropna=False,
      height=1.0,
      aspect=1.5,
      margin_titles=True,
  )

  g.map_dataframe(subplot_fn, x="imagenet_error", y="MetricValue")

  g.set_titles(row_template="", col_template="{col_name} varies")

  for ax in g.axes.flat:
    plotting.show_spines(ax)
    # Limits:
    ax.set_xlim(0.1, 0.35)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
    if plotting.row_num(ax) < 2:
      ax.set_ylim(0, 0.05)

    # Labels:
    if ax.is_first_col():
      if plotting.row_num(ax) == 0: ax.set_ylabel("ECE\n(temp.-scaled)")
      if plotting.row_num(ax) == 1: ax.set_ylabel("ECE\n(temp.-scaled)")

    if ax.is_last_row():
      ax.set_xlabel("Classification error")
    ax.grid(True, axis="both", zorder=-2000)

    if upstream_datasets[plotting.row_num(ax)] == "jft":
      ax.text(0.105, 0.002, "Pretraining: JFT", ha="left", va="bottom",
              color="gray", fontsize=5)
    if upstream_datasets[plotting.row_num(ax)] == "imagenet21k":
      ax.text(0.105, 0.002, "Pretraining: ImageNet-21k", ha="left",
              va="bottom", color="gray", fontsize=5)

    # Legend:
    if ax.is_first_row():
      _add_legend(ax, df_plot)

  plotting.apply_to_fig_text(g.fig, display.prettify)
  g.fig.tight_layout()
  return g


def plot_with_confidence(
    df_main: pd.DataFrame,
    upstream_dataset: str = "jft",
    gce_prefix: str = plotting.STD_GCE_PREFIX) -> sns.axisgrid.FacetGrid:
  """Plots acc/calib for ImageNet-C."""
  df_plot = _get_data(df_main, [upstream_dataset], gce_prefix)
  rescaling_methods = ["none", "temperature_scaling", "tau"]

  sns.set_style("ticks")
  g = sns.FacetGrid(
      data=df_plot,
      row="rescaling_method",
      row_order=rescaling_methods,
      col="varying_key",
      col_order=["size", "steps"],
      sharex=True,
      sharey=False,
      dropna=False,
      height=1.0,
      aspect=1.5,
      margin_titles=True,
  )

  g.map_dataframe(subplot_fn, x="imagenet_error", y="MetricValue")

  g.set_titles(row_template="", col_template="{col_name} varies")

  for ax in g.axes.flat:
    plotting.show_spines(ax)
    # Limits:
    ax.set_xlim(0.15, 0.35)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
    ax.set_ylim(0, 0.05)

    # Labels:
    if ax.is_first_col():
      if rescaling_methods[plotting.row_num(ax)] == "none":
        ax.set_ylabel(display.YLABEL_ECE_UNSCALED)
      if rescaling_methods[plotting.row_num(ax)] == "temperature_scaling":
        ax.set_ylabel(display.YLABEL_ECE_TEMP_SCALED_SHORT)
      if rescaling_methods[plotting.row_num(ax)] == "tau":
        ax.set_ylabel(display.YLABEL_TEMP_FACTOR_SHORT)

    if ax.is_last_row():
      ax.set_xlabel("Classification error")
    ax.grid(True, axis="both", zorder=-2000)

    # Legend:
    if ax.is_first_row():
      _add_legend(ax, df_plot)

  g.fig.suptitle(f"Pretraining dataset: {display.prettify(upstream_dataset)}",
                 x=0.55)

  plotting.apply_to_fig_text(g.fig, display.prettify)
  g.fig.tight_layout()

  for ax in g.axes.flat:
    if rescaling_methods[plotting.row_num(ax)] == "tau":
      ax.set_ylim(0.85, 1.05)
      ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
      plotting.annotate_confidence_plot(ax)
  return g


def _add_legend(ax, df_plot):
  """Add legend to size/steps plots."""
  scatter_objects = plotting.find_path_collection(ax, ["size", "steps"])
  varying_key = scatter_objects[0].get_label()
  vals = np.unique(df_plot[varying_key])
  formatter = mpl.ticker.FuncFormatter(
      lambda x, pos: f"{x/1e3:1.0f}k" if x < 1e6 else f"{x/1e6:1.0f}M")
  handles, labels = scatter_objects[0].legend_elements(
      prop="colors",
      func=lambda cs: np.array([vals[int(c)] for c in cs]),
      fmt=formatter,
      size=3)
  legend = ax.legend(
      handles,
      labels,
      bbox_to_anchor=(0, 1.04, 1, 0.2),
      loc="lower left",
      mode="expand",
      borderaxespad=0,
      ncol=len(labels),
      handletextpad=-0.5)
  legend.get_frame().set_linewidth(mpl.rcParams["axes.linewidth"])
  legend.get_frame().set_edgecolor("lightgray")
  ax.set_title(ax.get_title(), pad=15)


def _get_data(
    df_main: pd.DataFrame,
    upstream_datasets: Sequence[str],
    gce_prefix: str = plotting.STD_GCE_PREFIX) -> pd.DataFrame:
  """Get df_plot."""
  rescaling_methods = ["none", "temperature_scaling"]

  # Select data:
  mask = df_main.Metric.str.startswith(gce_prefix)
  mask &= df_main.rescaling_method.isin(rescaling_methods)
  mask &= df_main.DatasetName == "imagenet(split='validation[20%:]')"
  mask &= df_main.ModelName.str.contains("size")
  mask &= df_main.ModelName.str.contains("steps")
  df_plot = df_main[mask].copy()

  # Extract and insert size steps etc:
  pattern = re.compile(r"r\d+-x\d")
  df_plot["architecture"] = df_plot.ModelName.map(
      lambda s: utils.re_with_default(pattern, s))
  pattern = re.compile(r"(imagenet|imagenet21k|jft)(?=-)")
  df_plot["upstream_dataset"] = df_plot.ModelName.map(
      lambda s: utils.re_with_default(pattern, s))
  df_plot["size"] = df_plot.ModelName.map(
      lambda s: float(utils.re_with_default(r"(?<=size=)(\d+)", s, np.nan)))
  df_plot["steps"] = df_plot.ModelName.map(
      lambda s: float(utils.re_with_default(r"(?<=steps=)(\d+)", s, np.nan)))
  mask = df_plot.architecture.isin(["r50-x1", "r101-x3"])
  mask &= df_plot.upstream_dataset.isin(upstream_datasets)
  mask &= df_plot.steps != 457032  # Remove outlier run.
  df_plot = df_plot[mask].copy()

  # Add "optimal temperature" as another rescaling method, so that seaborn can
  # plot it as a third row:
  df_tau = df_plot[df_plot.rescaling_method == "temperature_scaling"].copy()
  df_tau.rescaling_method = "tau"
  df_tau.MetricValue = df_tau.tau_on_eval_data
  df_plot = pd.concat([df_plot, df_tau])
  rescaling_methods += ["tau"]

  # Add row that allows us to plot two representative conditions:
  df_cond1 = df_plot[df_plot["size"] == 13000000].copy()
  df_cond1["varying_key"] = "steps"
  df_cond2 = df_plot[df_plot["steps"] == 1120000].copy()
  df_cond2["varying_key"] = "size"
  return pd.concat([df_cond1, df_cond2])


def subplot_fn(data, x, y, **kwargs):
  """Seaborn-compatible subplot function."""
  del kwargs
  ax = plt.gca()
  varying_key = np.unique(data["varying_key"])
  assert len(varying_key) == 1
  varying_key = varying_key[0]
  utils.assert_no_duplicates_in_condition(
      data, ["architecture", varying_key])
  conditions = np.unique(data[varying_key]).tolist()
  if varying_key == "size":
    palette = "flare"
  elif varying_key == "steps":
    palette = "crest"
  else:
    palette = "flare"
  cmap = sns.color_palette(palette, n_colors=len(conditions), as_cmap=True)
  architectures = np.unique(data.architecture)
  for arch, markersize in zip(architectures, [30, 10]):
    data_sub = data[data.architecture == arch]
    data_sub = data_sub.sort_values(by=varying_key)
    colors = [conditions.index(cond) for cond in data_sub[varying_key]]
    try:
      ax.scatter(data_sub[x], data_sub[y], s=markersize, c=colors, cmap=cmap,
                 alpha=0.7, zorder=30, label=varying_key, linewidth=0)
    except:
      raise ValueError(data_sub)
    # White dots to mask lines:
    ax.scatter(data_sub[x], data_sub[y], s=markersize, c="w", alpha=1.0,
               zorder=20, linewidth=1.5)
    # Lines to connect dots:
    ax.plot(data_sub[x], data_sub[y], "-", color="gray", alpha=0.7, zorder=10,
            linewidth=0.75)
