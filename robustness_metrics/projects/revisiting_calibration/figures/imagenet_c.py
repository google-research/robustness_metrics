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

"""Figures for 'Revisiting Calibration of Modern Neural Networks'.

This module contains figures showing how ECE and accuracy behave under
distribution shift using ImageNet-C.
"""
from typing import List, Optional

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from robustness_metrics.projects.revisiting_calibration import display
from robustness_metrics.projects.revisiting_calibration import plotting
from robustness_metrics.projects.revisiting_calibration import utils
import seaborn as sns


def plot(df_main: pd.DataFrame,
         gce_prefix: str = plotting.STD_GCE_PREFIX,
         rescaling_methods: Optional[List[str]] = None,
         family_order: Optional[List[str]] = None,
         plot_confidence: bool = False,
         add_legend: bool = True,
         add_metric_description: bool = False,
         ) -> sns.axisgrid.FacetGrid:
  """Plots accuracy and calibration for ImageNet-C."""
  # Settings:
  rescaling_methods = rescaling_methods or ["none", "temperature_scaling"]
  if plot_confidence:
    rescaling_methods += ["tau"]
  family_order = family_order or ["mixer", "vit", "bit", "simclr", "wsl"]

  df_plot = _get_data(df_main, gce_prefix, family_order)

  # Add "optimal temperature" as another rescaling method, so that seaborn can
  # plot it as a third row:
  df_plot = plotting.add_optimal_temperature_as_rescaling_method(df_plot)

  df_plot["model_size"] = df_plot.ModelName.map(display.MODEL_SIZE)

  def subplot_fn(data, x, y, **kwargs):
    del kwargs
    ax = plt.gca()
    data = utils.average_imagenet_c_corruption_types(
        data, group_by=["severity", "model_size"])
    cmap = sns.color_palette(
        "flare", n_colors=data.severity.nunique(), as_cmap=True)
    # Plot data:
    ax.scatter(
        data[x], data[y],
        s=plotting.model_to_scatter_size(data.model_size, 0.75),
        c=data.severity,
        linewidth=0,
        cmap=cmap, alpha=0.7, zorder=30, label="data")
    # White dots to mask lines:
    ax.scatter(
        data[x], data[y],
        s=plotting.model_to_scatter_size(data.model_size, 0.75),
        c="w", alpha=1.0, zorder=20, linewidth=1.5, label="_hidden")
    # Lines to connect dots:
    for condition in np.unique(data["severity"]):
      data_sub = data[data["severity"] == condition].copy()
      data_sub = data_sub.sort_values(by=data_sub.columns.to_list())
      ax.plot(
          data_sub[x], data_sub[y],
          "-", color="gray", alpha=0.7, zorder=10, linewidth=1)

  g = plotting.FacetGrid(
      data=df_plot,
      sharex=True,
      sharey=False,
      dropna=False,
      row="rescaling_method",
      row_order=rescaling_methods,
      col="ModelFamily",
      col_order=family_order,
      height=1,
      aspect=1,
      margin_titles=True)

  g.map_dataframe(subplot_fn, x="downstream_error", y="MetricValue")

  g.set_titles(col_template="{col_name}", row_template="",
               size=mpl.rcParams["axes.titlesize"])

  # Format axes:
  for ax in g.axes.flat:
    if rescaling_methods[plotting.row_num(ax)] == "none":
      ax.set_ylim(0, 0.25)
      ax.set_yticks(np.arange(0, 0.272, 0.05))
      if plotting.is_first_col(ax):
        ax.set_ylabel(display.YLABEL_ECE_UNSCALED)

    elif rescaling_methods[plotting.row_num(ax)] == "temperature_scaling":
      ax.set_ylim(0, 0.15)
      ax.set_yticks(np.arange(0, 0.172, 0.05))
      if plotting.is_first_col(ax):
        ax.set_ylabel(display.YLABEL_ECE_TEMP_SCALED)

    elif rescaling_methods[plotting.row_num(ax)] == "tau":
      ax.set_ylim(0.6, 1.6)
      plotting.annotate_confidence_plot(ax)
      if plotting.is_first_col(ax):
        ax.set_ylabel(display.YLABEL_TEMP_FACTOR)

    ax.set_xlim(0.0, 0.8)
    ax.set_xticks(np.arange(0, 0.81, 0.2))
    plotting.show_spines(ax)
    if plotting.is_last_row(ax):
      ax.set_xlabel(display.XLABEL_INET_C_ERROR)
    ax.grid(axis="both")

    if not plotting.is_first_col(ax):
      ax.set_yticklabels("")

  g.fig.tight_layout()

  # Severity legend at bottom:
  if add_legend:
    scatter_objects = plotting.find_path_collection(
        g.axes.flat[1], label="data")
    handles, labels = scatter_objects[0].legend_elements(prop="colors")
    severity_legend = plt.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        title="ImageNet-C corruption severity",
        bbox_to_anchor=(0.5, 0),
        frameon=True,
        bbox_transform=g.fig.transFigure,
        ncol=6,
        handletextpad=0.1)
    severity_legend.get_frame().set_linewidth(mpl.rcParams["axes.linewidth"])
    severity_legend.get_frame().set_edgecolor("lightgray")
    g.fig.add_artist(severity_legend)

  if add_metric_description:
    plotting.add_metric_description_title(df_plot, g.fig)

  plotting.apply_to_fig_text(g.fig, display.prettify)

  return g


def _get_data(df_main: pd.DataFrame, gce_prefix: str,
              family_order: List[str]) -> pd.DataFrame:
  """Selects plotting data from df_main."""
  mask = df_main.Metric.str.startswith(gce_prefix)
  mask &= df_main.ModelName.isin(display.get_standard_model_list())
  mask &= df_main.rescaling_method.isin(["none", "temperature_scaling"])
  mask &= df_main.ModelFamily.isin(family_order)

  mask_before_dataset = mask.copy()
  mask &= df_main.DatasetName.str.startswith("imagenet_c")
  df_plot = df_main[mask].copy()

  # Add clean imagenet as severity zero:
  mask = mask_before_dataset & (df_main.DatasetName ==
                                "imagenet(split='validation[20%:]')")
  df_zero = df_main[mask].copy()
  df_zero.severity = 0
  df_plot = pd.concat([df_plot, df_zero])

  df_plot["model_size"] = df_plot.ModelName.map(display.MODEL_SIZE)
  return df_plot.copy()


def plot_error_increase_vs_model_size(
    df_main: pd.DataFrame,
    compact_layout: bool = True,
    gce_prefix=plotting.STD_GCE_PREFIX) -> sns.axisgrid.FacetGrid:
  """Plots acc/calib for ImageNet-C."""
  # Settings:
  rescaling_methods = ["temperature_scaling"]
  if compact_layout:
    family_order = ["mixer", "vit", "bit"]
  else:
    family_order = display.get_model_families_sorted()
    family_order.remove("alexnet")  # Remove AlexNet bc. it only has one size.

  df_plot = _get_data(df_main, gce_prefix, family_order)
  df_plot = df_plot[df_plot.rescaling_method.isin(rescaling_methods)].copy()

  # Add downstream_error as another "rescaling method", so that seaborn can
  # plot it as a separate row:
  df_tau = df_plot.copy()
  df_tau.rescaling_method = "downstream_error"
  df_tau.MetricValue = df_tau.downstream_error
  df_plot = pd.concat([df_plot, df_tau])
  rescaling_methods = ["temperature_scaling", "downstream_error"]

  df_plot = utils.average_imagenet_c_corruption_types(
      df_plot, group_by=["ModelName", "Metric", "severity", "rescaling_method"])

  # Normalize per severity:
  rescaling_methods = df_plot.rescaling_method.unique()
  datasets = df_plot.DatasetName.unique()
  df_plot["relative_metric_value"] = np.nan
  for family in family_order:
    for severity in df_plot.severity.unique():
      for method in rescaling_methods:
        for dataset_ in datasets:
          mask = df_plot.severity == severity
          mask &= df_plot.rescaling_method == method
          mask &= df_plot.DatasetName == dataset_
          mask &= df_plot.ModelFamily == family
          df_masked = df_plot[mask].copy()
          if not df_masked.shape[0]:
            continue
          size_mask = df_masked.model_size == df_masked.model_size.max()
          largest_model_value = float(df_masked.loc[size_mask, "MetricValue"])
          df_plot.loc[mask, "relative_metric_value"] = (df_masked.MetricValue -
                                                        largest_model_value)

  cmap = sns.color_palette(
      "flare", n_colors=df_plot.severity.nunique(), as_cmap=True)

  def subplot_fn(data, x, y, **kwargs):
    del kwargs
    ax = plt.gca()
    for condition in np.unique(data["model_size"]):
      data_sub = data[data["model_size"] == condition].copy()
      data_sub = data_sub.sort_values(by=data_sub.columns.to_list())
      # Plot data:
      ax.scatter(
          data_sub[x], data_sub[y],
          s=plotting.model_to_scatter_size(data_sub.model_size, 0.75),
          c=data_sub.severity, cmap=cmap, alpha=0.7, zorder=30, label="data",
          vmin=0, vmax=5, linewidth=0)
      # White dots to mask lines:
      ax.scatter(
          data_sub[x], data_sub[y],
          s=plotting.model_to_scatter_size(data_sub.model_size, 0.75),
          c="w", alpha=1.0, zorder=20, linewidth=1.5, label="_hidden")
      # Lines to connect dots:
      ax.plot(
          data_sub[x], data_sub[y], "-", color="gray", alpha=0.7, zorder=10,
          linewidth=0.75)

  g = plotting.FacetGrid(
      data=df_plot,
      sharex=True,
      sharey=False,
      dropna=False,
      col="ModelFamily",
      col_order=family_order,
      row="rescaling_method",
      row_order=rescaling_methods,
      height=1.1,
      margin_titles=True,
      aspect=0.8)
  g.map_dataframe(subplot_fn, x="severity", y="relative_metric_value")
  g.set_titles(col_template="{col_name}", row_template="",
               size=mpl.rcParams["axes.titlesize"])

  # Format axes:
  for ax in g.axes.flatten():
    ax.set_xlim(-0.9, 5.9)
    plotting.show_spines(ax)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.02))
    if plotting.is_first_col(ax) and plotting.row_num(ax) == 0:
      ax.set_ylabel("Classification error\n(Δ to largest model)")
    elif plotting.is_first_col(ax) and plotting.row_num(ax) == 1:
      ax.set_ylabel("ECE\n(Δ to largest model)")
    else:
      ax.set_yticklabels([])

    if plotting.is_first_row(ax):
      ax.set_title(family_order[plotting.col_num(ax)])
      ax.set_ylim(-0.05 if compact_layout else -0.1, 0.3)
      ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))

    if plotting.is_last_row(ax):
      ax.set_xlabel("Corruption\nseverity")
      ax.set_ylim(-0.01 if compact_layout else -0.04, 0.03)
      ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.02))
    ax.grid(axis="both")
    ax.grid(True, axis="y", which="minor")

  plotting.apply_to_fig_text(g.fig, display.prettify)
  g.fig.tight_layout()
  g.fig.subplots_adjust(wspace=0.15)
  return g
