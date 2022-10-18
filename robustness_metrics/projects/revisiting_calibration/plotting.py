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

"""Low-level plotting code for 'Revisiting Calibration'."""

import os
import time
from typing import Callable, List, Optional, Union

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from robustness_metrics.projects.revisiting_calibration import utils
import seaborn as sns


STD_GCE_PREFIX = (
    "gce(binning_scheme='adaptive',max_prob=True,class_conditional=False,"
    "norm='l1',num_bins=100,threshold=0.0")


def set_seaborn_theme():
  """Sets seaborn theme with customizations for "Revisiting Calibration"."""
  rc = {}

  rc["axes.linewidth"] = 0.4
  rc["axes.edgecolor"] = "k"
  rc["axes.labelpad"] = 1
  rc["axes.titlesize"] = 8
  rc["axes.titlepad"] = 3

  rc["lines.linewidth"] = 0.5

  rc["grid.linewidth"] = 0.25
  rc["legend.fontsize"] = 5
  rc["scatter.edgecolors"] = "face"

  ticksize = 2
  rc["xtick.major.size"] = ticksize
  rc["xtick.minor.size"] = ticksize / 2
  rc["ytick.major.size"] = ticksize
  rc["ytick.minor.size"] = ticksize / 2

  tickwidth = rc["axes.linewidth"]
  rc["xtick.major.width"] = tickwidth
  rc["xtick.minor.width"] = tickwidth
  rc["ytick.major.width"] = tickwidth
  rc["ytick.minor.width"] = tickwidth

  tickpad = 1
  rc["xtick.major.pad"] = tickpad
  rc["ytick.major.pad"] = tickpad

  sns.set_theme(context="paper", style="ticks", font_scale=0.6, rc=rc)


class FacetGrid(sns.FacetGrid):
  """Custom wrapper around Seaborn's FacetGrid."""

  def tight_layout(self, *args, **kwargs):
    """Fixes an issue in FacetGrid when titles are too wide for margins.

    Seaborn FacetGrid calls Matplotlib's tight_layout automatically upon plot
    creation. If titles are very wide, tight_layout can cause the subplot width
    to become negative, which causes an error in Seaborn. We sometimes have long
    titles because plot titles are generated automatically from the dataframe
    columns. We shorten/prettify these titles manually, but only after the plot
    is created. By overriding tight_layout, we can create the plot without
    triggering the error and then call tight_layout manually as needed, after
    fixing the titles.

    Args:
      *args: Unused.
      **kwargs: Unused.
    """
    pass


def model_to_scatter_size(model_size: float, factor: float = 1.0) -> float:
  model_size = np.maximum(model_size, 1.5)
  return (2.0 * factor * model_size) ** 1.75


def show_spines(ax: mpl.axes.Axes):
  for spine in ax.spines.values():
    spine.set_visible(True)


def apply_to_fig_text(fig: mpl.figure.Figure, fn: Callable[[str], str]):
  """Applies `fn` to each `Text` object in `fig`. Use to prettify labels."""
  for text in fig.findobj(match=plt.Text):
    text.set_text(fn(text.get_text()))


def get_model_family_legend(ax: mpl.axes.Axes, family_order: List[str]):
  """Adds a legend for the model family markers and colors below the axes."""
  family_color_dict = dict(enumerate(family_order))
  handles = []
  labels = []
  for family in family_order:
    path_collections = find_path_collection(ax, label=family)
    if not path_collections:
      continue
    handle, legend = path_collections[0].legend_elements(
        prop="colors",
        fmt=mpl.ticker.FuncFormatter(lambda color, _: family_color_dict[color]),
        size=4)
    handles.extend(handle)
    labels.extend(legend)
  return handles, labels


def find_path_collection(ax: mpl.axes.Axes, label: Union[str, List[str]]):
  """Finds path collections in axes, for creating legends."""
  if isinstance(label, str):
    label = [label]
  def match_fn(x):
    return (isinstance(x, mpl.collections.PathCollection) and
            (x.get_label() in label))
  return ax.findobj(match=match_fn)


def annotate_confidence_plot(ax: mpl.axes.Axes):
  """Adds annotations to indicate over/underconfidence."""
  line = ax.axhline(1.0, linestyle="--", color="k", zorder=1)
  disp_coords = line.get_transform().transform([0, line.get_ydata()[0]])
  line_y_ax = ax.transAxes.inverted().transform(disp_coords)[1]
  offset = 0.02
  shared_kws = {"color": "gray", "zorder": 2, "transform": ax.transAxes,
                "fontsize": 4}
  ax.text(1.0 - offset, line_y_ax + offset, "overconfident", va="bottom",
          ha="right", **shared_kws)
  ax.text(1.0 - offset, line_y_ax - offset, "underconfident", va="top",
          ha="right", **shared_kws)


def save_fig(fig: mpl.figure.Figure,
             filename: str,
             directory: Optional[str] = None,
             add_date: bool = False,
             **savefig_kwargs):
  """Saves a Matplotlib figure to disk."""
  if add_date:
    filename = "{}_{}".format(time.strftime("%Y%m%d"), filename)
  savefig_kwargs.setdefault("bbox_inches", "tight")
  savefig_kwargs.setdefault("pad_inches", 0.02)
  save_path = os.path.join(directory, filename)
  fig.savefig(save_path, **savefig_kwargs)


def add_metric_description_title(df_plot: pd.DataFrame,
                                 fig: mpl.figure.Figure,
                                 y: float = 1.0):
  """Adds a suptitle to the figure, describing the metric used."""
  assert df_plot.Metric.nunique() == 1, "More than one metric in DataFrame."
  binning_scheme = utils.assert_and_get_constant(df_plot.binning_scheme)
  num_bins = utils.assert_and_get_constant(df_plot.num_bins)
  norm = utils.assert_and_get_constant(df_plot.norm)
  title = (f"ECE variant: {binning_scheme} binning, "
           f"{num_bins:.0f} bins, "
           f"{norm} norm")
  display_names = {
      "adaptive": "equal-mass",
      "even": "equal-width",
      "l1": "L1",
      "l2": "L2",
  }
  for old, new in display_names.items():
    title = title.replace(old, new)

  fig.suptitle(title, y=y, verticalalignment="bottom")


def add_optimal_temperature_as_rescaling_method(
    df_plot: pd.DataFrame) -> pd.DataFrame:
  """Adds "optimal temperature" as another rescaling method to df_plot."""
  df_tau = df_plot[df_plot.rescaling_method == "temperature_scaling"].copy()
  df_tau.rescaling_method = "tau"
  df_tau.MetricValue = df_tau.tau_on_eval_data
  return pd.concat([df_plot, df_tau])


def row_num(ax: mpl.axes.Axes) -> int:
  """Gets the row number of a subplot axis in a figure."""
  return ax.get_subplotspec().rowspan.start


def col_num(ax: mpl.axes.Axes) -> int:
  """Gets the column number of a subplot axis in a figure."""
  return ax.get_subplotspec().colspan.start


def is_first_row(ax: mpl.axes.Axes) -> bool:
  """Returns if axis is in first row of a subplot."""
  try:
    return ax.is_first_row()
  except AttributeError:
    # matplotlib 3.6
    return ax.get_subplotspec().is_first_row()


def is_first_col(ax: mpl.axes.Axes) -> bool:
  """Returns if axis is in first col of a subplot."""
  try:
    return ax.is_first_col()
  except AttributeError:
    # matplotlib 3.6
    return ax.get_subplotspec().is_first_col()


def is_last_row(ax: mpl.axes.Axes) -> bool:
  """Returns if axis is in last row of a subplot."""
  try:
    return ax.is_last_row()
  except AttributeError:
    # matplotlib 3.6
    return ax.get_subplotspec().is_last_row()


def is_last_col(ax: mpl.axes.Axes) -> bool:
  """Returns if axis is in last col of a subplot."""
  try:
    return ax.is_last_col()
  except AttributeError:
    # matplotlib 3.6
    return ax.get_subplotspec().is_last_col()
