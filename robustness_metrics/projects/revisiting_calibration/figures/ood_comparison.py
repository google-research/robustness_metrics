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

This module contains figures showing ECE and accuracy on several out-of-
distribution benchmark datasets.
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
from sklearn import linear_model


def plot(df_main: pd.DataFrame,
         gce_prefix: str = plotting.STD_GCE_PREFIX,
         rescaling_methods: Optional[List[str]] = None,
         plot_confidence: bool = False,
         add_legend: bool = True,
         legend_position: str = "right",
         height: float = 1.05,
         aspect: float = 1.2,
         add_metric_description: bool = False,
         ) -> sns.axisgrid.FacetGrid:
  """Plots acc/calib for ImageNet and natural OOD datasets."""
  # Settings:
  rescaling_methods = rescaling_methods or ["none", "temperature_scaling"]
  if plot_confidence:
    rescaling_methods += ["tau"]
  family_order = display.get_model_families_sorted()
  dataset_order = display.OOD_DATASET_ORDER
  # Select data:
  mask = df_main.Metric.str.startswith(gce_prefix)
  mask &= df_main.ModelName.isin(display.get_standard_model_list())
  mask &= df_main.DatasetName.isin(dataset_order)
  mask &= df_main.rescaling_method.isin(rescaling_methods)
  mask &= ~(df_main.DatasetName.str.startswith("imagenet_a") &
            (~df_main.use_dataset_labelset.eq(True)))
  mask &= ~(df_main.DatasetName.str.startswith("imagenet_r") &
            (~df_main.use_dataset_labelset.eq(True)))
  family_order = display.get_model_families_sorted()
  mask &= df_main.ModelFamily.isin(family_order)
  mask &= df_main.DatasetName.isin(dataset_order)
  df_plot = df_main[mask].copy()
  df_plot, cmap = display.add_display_data(df_plot, family_order)

  # Remove "use_dataset_labelset=True" to have uniform metric name:
  df_plot.Metric = df_plot.Metric.str.replace("use_dataset_labelset=True,", "")

  # Add "optimal temperature" as another rescaling method, so that seaborn can
  # plot it as a third row:
  df_tau = df_plot[df_plot.rescaling_method == "temperature_scaling"].copy()
  df_tau.rescaling_method = "tau"
  df_tau.MetricValue = df_tau.tau_on_eval_data
  df_plot = pd.concat([df_plot, df_tau])

  def subplot_fn(data, x, y, **kwargs):
    del kwargs
    ax = plt.gca()
    for marker in data.family_marker.unique():
      data_sub = data[data.family_marker == marker]
      ax.scatter(
          data_sub[x],
          data_sub[y],
          s=plotting.model_to_scatter_size(data_sub.model_size),
          c=data_sub.family_index,
          cmap=cmap,
          vmin=0,
          vmax=len(family_order),
          marker=marker,
          linewidth=0,
          alpha=0.7,
          zorder=30,
          label=utils.assert_and_get_constant(data_sub.ModelFamily))

  g = plotting.FacetGrid(
      data=df_plot,
      sharex=False,
      sharey=False,
      dropna=False,
      col="DatasetName",
      col_order=dataset_order,
      row="rescaling_method",
      row_order=rescaling_methods,
      height=height,
      aspect=aspect,
      margin_titles=True)

  g.map_dataframe(subplot_fn, x="downstream_error", y="MetricValue")

  g.set_titles(col_template="{col_name}", row_template="",
               size=mpl.rcParams["axes.titlesize"])

  for ax in g.axes.flat:
    plotting.show_spines(ax)
    ax.grid(True, axis="both")
    ax.grid(True, axis="both", which="minor")

    ax.set_xlim(left=0.1)
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    if rescaling_methods[plotting.row_num(ax)] != "tau":
      ax.set_yticks([], minor=True)
      ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.02))
      if dataset_order[plotting.col_num(
          ax)] == "imagenet(split='validation[20%:]')":
        if rescaling_methods[plotting.row_num(ax)] == "none":
          ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.04))
      if dataset_order[plotting.col_num(
          ax)] == "imagenet_v2(variant='MATCHED_FREQUENCY')":
        ax.set_ylim(top=0.14)
        if rescaling_methods[plotting.row_num(ax)] == "none":
          ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.02))
        if rescaling_methods[plotting.row_num(ax)] == "temperature_scaling":
          ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.02))
      if dataset_order[plotting.col_num(ax)] == "imagenet_r":
        ax.set_ylim(top=0.3)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))

    if dataset_order[plotting.col_num(
        ax)] == "imagenet_v2(variant='MATCHED_FREQUENCY')":
      ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    if dataset_order[plotting.col_num(ax)] == "imagenet_a":
      if rescaling_methods[plotting.row_num(ax)] != "tau":
        ax.set_ylim(top=0.6)
      ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))

    # Labels:
    if ax.is_first_col():
      if rescaling_methods[plotting.row_num(ax)] == "none":
        ax.set_ylabel(display.YLABEL_ECE_UNSCALED)
      elif rescaling_methods[plotting.row_num(ax)] == "temperature_scaling":
        ax.set_ylabel(display.YLABEL_ECE_TEMP_SCALED)
      elif rescaling_methods[plotting.row_num(ax)] == "beta_scaling":
        ax.set_ylabel(display.YLABEL_ECE_BETA_SCALED)
      elif rescaling_methods[plotting.row_num(ax)] == "tau":
        ax.set_ylabel(display.YLABEL_TEMP_FACTOR)
      else:
        ax.set_ylabel(rescaling_methods[plotting.row_num(ax)])

    if ax.is_last_row():
      ax.set_xlabel(display.XLABEL_CLASSIFICATION_ERROR)
    else:
      ax.set_xticklabels("")

  if add_metric_description:
    plotting.add_metric_description_title(df_plot, g.fig, y=1.05)

  plotting.apply_to_fig_text(g.fig, display.prettify)
  g.fig.tight_layout(pad=0)
  g.fig.subplots_adjust(wspace=0.2, hspace=0.2)

  for ax in g.axes.flat:
    if rescaling_methods[plotting.row_num(ax)] == "tau":
      ax.set_ylim(0.5, 2.5)
      plotting.annotate_confidence_plot(ax)

  if add_legend:
    handles, labels = plotting.get_model_family_legend(
        g.axes.flat[0], family_order)
    if legend_position == "below":
      # Model family legend below plot:
      legend = g.axes.flat[0].legend(
          handles=handles, labels=labels, loc="upper center",
          title="Model family", bbox_to_anchor=(0.5, 0.0), frameon=True,
          bbox_transform=g.fig.transFigure, ncol=len(family_order),
          handletextpad=0.1)
    elif legend_position == "right":
      # Model family legend next to plot:
      legend = g.axes.flat[0].legend(
          handles=handles, labels=labels, loc="center left",
          title="Model family", bbox_to_anchor=(1.00, 0.53), frameon=True,
          bbox_transform=g.fig.transFigure, ncol=1,
          handletextpad=0.1)
    legend.get_frame().set_linewidth(mpl.rcParams["axes.linewidth"])
    legend.get_frame().set_edgecolor("lightgray")
    plotting.apply_to_fig_text(g.fig, display.prettify)
  return g


def plot_alternative_metrics(
    df_main: pd.DataFrame,
    xs: List[str], ys: List[str],
    gce_prefix: str = plotting.STD_GCE_PREFIX,
    add_legend: bool = True,
    add_metric_description: bool = False,
) -> sns.axisgrid.FacetGrid:
  """Plots acc/calib for ImageNet and natural OOD datasets."""
  # Settings:
  rescaling_method = "temperature_scaling"
  family_order = display.get_model_families_sorted()
  dataset_order = (["imagenet(split='validation[20%:]')"] +
                   display.OOD_DATASET_ORDER)

  # Select data:
  mask = df_main.Metric.str.startswith(gce_prefix)
  mask &= df_main.ModelName.isin(display.get_standard_model_list())
  mask &= df_main.DatasetName.isin(dataset_order)
  mask &= df_main.rescaling_method.isin([rescaling_method])
  mask &= ~(df_main.DatasetName.str.startswith("imagenet_a") &
            (~df_main.use_dataset_labelset.eq(True)))
  mask &= ~(df_main.DatasetName.str.startswith("imagenet_r") &
            (~df_main.use_dataset_labelset.eq(True)))
  family_order = display.get_model_families_sorted()
  mask &= df_main.ModelFamily.isin(family_order)
  mask &= df_main.DatasetName.isin(dataset_order)
  df_plot = df_main[mask].copy()
  df_plot, cmap = display.add_display_data(df_plot, family_order)

  # Remove "use_dataset_labelset=True" to have uniform metric name:
  df_plot.Metric = df_plot.Metric.str.replace("use_dataset_labelset=True,", "")

  # Repeat dataframe with rows:
  df_orig = df_plot
  df_plot = pd.DataFrame()
  assert not ((len(set(xs)) > 1) and (len(set(ys)) > 1)), (
      "One of x and y must contain a constant value.")
  varying_set = xs if (len(set(xs)) > 1) else ys
  for x, y, varying in zip(xs, ys, varying_set):
    df_here = df_orig.copy()
    df_here["x"] = x
    df_here["y"] = y
    df_here["row"] = varying
    df_plot = pd.concat([df_plot, df_here])

  def get_residual(classification_error, metric):
    reg = linear_model.LinearRegression()
    metric = metric.to_numpy()
    not_nan = ~np.isnan(metric)
    x = classification_error.to_numpy()[:, None]
    reg.fit(x[not_nan, :], metric[not_nan])
    return metric - reg.predict(x)

  def subplot_fn(data, **kwargs):
    del kwargs
    ax = plt.gca()
    x = ax.my_xlabel = utils.assert_and_get_constant(data.x)
    y = ax.my_ylabel = utils.assert_and_get_constant(data.y)
    if y.endswith("_residual"):
      # Plot residual w.r.t. accuracy:
      y = y.replace("_residual", "")
      data = data.copy()
      data.loc[:, y] = get_residual(data["downstream_error"], data[y])

    for marker in data.family_marker.unique():
      data_sub = data[data.family_marker == marker]
      ax.scatter(
          data_sub[x],
          data_sub[y],
          s=plotting.model_to_scatter_size(data_sub.model_size),
          c=data_sub.family_index,
          cmap=cmap,
          vmin=0,
          vmax=len(family_order),
          marker=marker,
          alpha=0.7,
          zorder=30,
          label=utils.assert_and_get_constant(data_sub.ModelFamily),
          linewidth=0.0)

  g = plotting.FacetGrid(
      data=df_plot,
      sharex=False,
      sharey=False,
      dropna=False,
      col="DatasetName",
      col_order=dataset_order,
      row="row",
      row_order=varying_set,
      height=1.05,
      aspect=1.3,
      margin_titles=True)

  g.map_dataframe(subplot_fn)

  g.set_titles(col_template="{col_name}", row_template="",
               size=mpl.rcParams["axes.titlesize"])

  for ax in g.axes.flat:
    plotting.show_spines(ax)
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))

    col = plotting.col_num(ax)
    if dataset_order[col] == "imagenet_v2(variant='MATCHED_FREQUENCY')":
      ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    if dataset_order[col] == "imagenet_r":
      ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    if dataset_order[col] == "imagenet_a":
      ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))

    if varying_set == xs:
      ax.set_xlabel(ax.my_xlabel)
      if ax.is_first_col():
        ax.set_ylabel(ax.my_ylabel)
    elif varying_set == ys:
      if ax.is_last_row():
        ax.set_xlabel(ax.my_xlabel)
      else:
        ax.set_xticklabels([])
      if ax.is_first_col():
        ax.set_ylabel(ax.my_ylabel)
    ax.grid(True, axis="x", which="minor")
    ax.grid(True, axis="both")

  if add_metric_description:
    plotting.add_metric_description_title(df_plot, g.fig, y=1.05)

  def prettify(x):
    x = display.prettify(x)
    if x == "downstream_error": return display.XLABEL_CLASSIFICATION_ERROR
    if x == "MetricValue": return display.YLABEL_ECE_TEMP_SCALED_SHORT
    if x == "brier": return "Brier score\n(temp.-scaled)"
    if x == "brier_div_error": return "Brier / class. error"
    if x == "brier_residual": return "Brier (residual)"
    if x == "nll": return "NLL\n(temp.-scaled)"
    if x == "nll_div_error": return "NLL / class. error"
    if x == "nll_residual": return "NLL (residual)"
    return x

  plotting.apply_to_fig_text(g.fig, prettify)
  g.fig.tight_layout()
  g.fig.subplots_adjust(wspace=0.4, hspace=0.1 if varying_set == ys else 0.5)

  # Model family legend below plot:
  if add_legend:
    handles, labels = plotting.get_model_family_legend(
        g.axes.flat[0], family_order)
    legend = g.axes.flat[0].legend(
        handles=handles, labels=labels, loc="upper center",
        title="Model family", bbox_to_anchor=(0.5, 0), frameon=True,
        bbox_transform=g.fig.transFigure, ncol=len(family_order),
        handletextpad=0.1)
    legend.get_frame().set_linewidth(mpl.rcParams["axes.linewidth"])
    legend.get_frame().set_edgecolor("lightgray")
    plotting.apply_to_fig_text(g.fig, prettify)

  return g
