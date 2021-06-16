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

This module contains figures showing regression fits between accuracy and
calibration, across dataset and model variants within a family.
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
from seaborn import algorithms as sns_algos
from seaborn import utils as sns_utils


def plot(df_main: pd.DataFrame,
         family_order: Optional[List[str]] = None,
         rescaling_method: str = "temperature_scaling",
         gce_prefix: str = plotting.STD_GCE_PREFIX) -> sns.axisgrid.FacetGrid:
  """Plots a regression of accuracy and calibration, split by model family."""
  # Settings:
  if family_order is None:
    family_order = display.get_model_families_sorted()
  dataset_order = [
      "imagenet(split='validation[20%:]')",
      "imagenet_v2(variant='MATCHED_FREQUENCY')",
      "imagenet_r",
      "imagenet_a",
  ]

  # Select data:
  mask = df_main.Metric.str.startswith(gce_prefix)
  mask &= df_main.ModelName.isin(display.get_standard_model_list())
  mask &= df_main.rescaling_method == rescaling_method
  mask &= df_main.ModelFamily.isin(family_order)
  mask &= ~(df_main.DatasetName.str.startswith("imagenet_a") &
            ~df_main.use_dataset_labelset.eq(True))
  mask &= ~(df_main.DatasetName.str.startswith("imagenet_r") &
            ~df_main.use_dataset_labelset.eq(True))
  df_plot = df_main[mask].copy()

  # Select OOD datasets:
  mask = df_plot.DatasetName.isin(dataset_order)
  mask |= df_plot.DatasetName.str.startswith("imagenet_c")
  df_plot = df_plot[mask].copy()

  # Add plotting-related data:
  df_plot, _ = display.add_display_data(df_plot, family_order)
  df_plot["dataset_group"] = "others"
  im_c_mask = df_plot.DatasetName.str.startswith("imagenet_c")
  df_plot.loc[im_c_mask, "dataset_group"] = "imagenet_c"
  dataset_group_order = ["imagenet_c", "others"]

  def subplot_fn(**kwargs):
    x = kwargs["x"]
    y = kwargs["y"]
    data = kwargs["data"]
    utils.assert_no_duplicates_in_condition(
        data, group_by=["DatasetName", "ModelName"])
    ax = plt.gca()
    kwargs["color"] = utils.assert_and_get_constant(data.family_color)
    ax = sns.regplot(ax=ax, **kwargs)
    plotter = RegressionPlotter(x, y, data=data)

    # Get regression parameters and show in plot:
    grid = np.linspace(data[x].min(), data[x].max(), 100)
    beta_plot, beta_boots = plotter.get_params(grid)
    beta_plot = np.array(beta_plot)
    beta_boots = np.array(beta_boots)
    intercept = 10 ** np.median(beta_boots[0, :])
    intercept_ci = 10 ** sns_utils.ci(beta_boots[0, :])
    slope = np.median(beta_boots[1, :])
    slope_ci = sns_utils.ci(beta_boots[1, :])
    s = (f"a = {intercept:1.2f} ({intercept_ci[0]:1.2f}, "
         f"{intercept_ci[1]:1.2f})\nk = {slope:1.2f} ({slope_ci[0]:1.2f}, "
         f"{slope_ci[1]:1.2f})")
    ax.text(0.04, 0.96, s, va="top", ha="left", transform=ax.transAxes,
            fontsize=4, color=(0.3, 0.3, 0.3),
            bbox=dict(facecolor="w", alpha=0.8, boxstyle="square,pad=0.1"))

  df_plot["MetricValue_log"] = np.log10(df_plot.MetricValue)
  df_plot["downstream_error_log"] = np.log10(df_plot.downstream_error)

  g = plotting.FacetGrid(
      data=df_plot,
      sharex=False,
      sharey=False,
      col="ModelFamily",
      col_order=family_order,
      hue="ModelFamily",
      hue_order=family_order,
      row="dataset_group",
      row_order=dataset_group_order,
      height=1.0,
      aspect=0.9,)

  g.map_dataframe(subplot_fn, x="downstream_error_log", y="MetricValue_log",
                  scatter_kws={"alpha": 0.5, "linewidths": 0.0, "s": 2})

  g.set_titles(template="{col_name}", size=mpl.rcParams["axes.titlesize"])

  for ax in g.axes.flat:
    plotting.show_spines(ax)
    ax.set_xlim(np.log10(0.1), np.log10(1.0))
    xticks = np.arange(0.1, 1.0 + 0.001, 0.1)
    ax.set_xticks(np.log10(xticks))
    if ax.is_last_row():
      show = [0.1, 0.2, 0.4, 1.0]
      xticklabels = [f"{x:0.1f}"if x in show else "" for x in xticks]
      ax.set_xticklabels(xticklabels)
      ax.set_title("")
    else:
      ax.set_xticklabels([])

    ax.set_ylim(np.log10(0.01), np.log10(0.8))
    yticks = np.arange(0.05, 0.8 + 0.001, 0.05)
    ax.set_yticks(np.log10(yticks))
    if ax.is_first_col():
      show = [0.1, 0.2, 0.4, 0.8]
      yticklabels = [f"{x:0.1f}"if x in show else "" for x in yticks]
      ax.set_yticklabels(yticklabels)
    else:
      ax.set_yticklabels([])

    ax.grid(True, axis="both")

    # Labels:
    if ax.is_last_row():
      ax.set_xlabel(display.XLABEL_CLASSIFICATION_ERROR)
    if ax.is_first_col():
      ax.set_ylabel(display.YLABEL_ECE_TEMP_SCALED_SHORT)
  plotting.apply_to_fig_text(g.fig, display.prettify)
  g.fig.tight_layout(w_pad=0)
  return g


class RegressionPlotter(sns.regression._RegressionPlotter):  # pylint: disable=protected-access
  """Adds ability to get regression parameters from Seaborn plotter."""

  def get_params(self, grid):
    """Low-level regression and prediction. Adapted from seaborn."""

    def reg_func(x_, y_):
      return np.linalg.pinv(x_).dot(y_)

    x, y = np.c_[np.ones(len(self.x)), self.x], self.y
    grid = np.c_[np.ones(len(grid)), grid]
    beta_plot = reg_func(x, y)
    yhat = grid.dot(beta_plot)
    if self.ci is None:
      return yhat, None
    beta_boots = sns_algos.bootstrap(
        x, y, func=reg_func, n_boot=self.n_boot, units=self.units,  # pytype: disable=attribute-error
        seed=self.seed).T
    return beta_plot, beta_boots
