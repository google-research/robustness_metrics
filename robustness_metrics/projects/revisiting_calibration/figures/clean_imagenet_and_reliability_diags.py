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

This module contains figures showing ECE and accuracy, as well as reliability
diagrams, on clean ImageNet.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from robustness_metrics.projects.revisiting_calibration import display
from robustness_metrics.projects.revisiting_calibration import plotting
from robustness_metrics.projects.revisiting_calibration import utils


def plot(df_main: pd.DataFrame,
         df_reliability: pd.DataFrame,
         gce_prefix: str = plotting.STD_GCE_PREFIX,
         rescaling_method: str = "temperature_scaling",
         add_guo: bool = False) -> mpl.figure.Figure:
  """Plots acc/calib and reliability diagrams on clean ImageNet (Figure 1)."""

  family_order = display.get_model_families_sorted()
  if add_guo:
    family_order.append("guo")

  if rescaling_method == "both":
    rescaling_methods = ["none", "temperature_scaling"]
  else:
    rescaling_methods = [rescaling_method]

  # Set up figure:
  fig = plt.figure(figsize=(display.FULL_WIDTH, 1.6))
  if rescaling_method == "both":
    widths = [1.75, 1.75, 1, 1, 1]
  else:
    widths = [1.8, 1, 1, 1, 1, 1]
  heights = [0.3, 1]
  spec = fig.add_gridspec(
      ncols=len(widths),
      nrows=len(heights),
      width_ratios=widths,
      height_ratios=heights)

  # First panels: acc vs calib ImageNet:
  for ax_i, rescaling_method in enumerate(rescaling_methods):

    df_plot, cmap = _get_data(df_main, gce_prefix, family_order,
                              rescaling_methods=[rescaling_method])
    ax = fig.add_subplot(
        spec[0:2, ax_i], box_aspect=1.0)
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
    ax.set_anchor("N")
    ax.grid(False, which="minor")
    ax.grid(True, axis="both")
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.01))
    ax.set_ylim(bottom=0.01, top=0.09)
    ax.set_xlim(0.05, 0.55)
    ax.set_xlabel(display.XLABEL_INET_ERROR)
    if len(rescaling_methods) == 1:  # Showing just one rescaling method.
      if rescaling_method == "none":
        ax.set_ylabel(display.YLABEL_ECE_UNSCALED)
      elif rescaling_method == "temperature_scaling":
        ax.set_ylabel(display.YLABEL_ECE_TEMP_SCALED)
    else:  # Showing both rescaling methods.
      if rescaling_method == "none":
        ax.set_title("Unscaled")
      elif rescaling_method == "temperature_scaling":
        ax.set_title("Temperature-scaled")
      if ax.is_first_col():
        ax.set_ylabel("ECE")

  # Remaining panels: Reliability diagrams:
  offset = len(rescaling_methods)
  model_names = [
      "mixer/jft-300m/H/14",
      "vit-h/14",
      "bit-jft-r152-x4-480",
      ]
  if offset == 1:
    model_names += ["wsl_32x48d", "simclr-4x-fine-tuned-100"]
  dataset_name = "imagenet(split='validation[20%:]')"

  for i, model_name in enumerate(model_names):
    # Get predictions:
    mask = df_main.ModelName == model_name
    mask &= df_main.rescaling_method == "none"
    mask &= df_main.Metric == "accuracy"
    mask &= df_main.DatasetName == dataset_name
    raw_model_name = df_main[mask].RawModelName
    assert len(raw_model_name) <= 1, df_main[mask]
    if len(raw_model_name) == 0:  # pylint: disable=g-explicit-length-test
      continue

    binned = _get_binned_reliability_data(
        df_reliability, utils.assert_and_get_constant(raw_model_name),
        dataset_name)

    rel_ax = fig.add_subplot(spec[1, i + offset])
    _plot_confidence_and_reliability(
        conf_ax=fig.add_subplot(spec[0, i + offset]),
        rel_ax=rel_ax, binned=binned, model_name=model_name, first_col=offset)

    if rescaling_method == "none":
      rel_ax.set_xlabel("Confidence\n(unscaled)")
    elif rescaling_method == "temperature_scaled":
      rel_ax.set_xlabel("Confidence\n(temp. scaled)")

  # Model family legend:
  handles, labels = plotting.get_model_family_legend(big_ax, family_order)
  legend = big_ax.legend(
      handles=handles, labels=labels, loc="upper right", frameon=True,
      labelspacing=0.25, handletextpad=0.1, borderpad=0.3, fontsize=4)
  legend.get_frame().set_linewidth(mpl.rcParams["axes.linewidth"])
  legend.get_frame().set_edgecolor("lightgray")

  plotting.apply_to_fig_text(fig, display.prettify)

  offset = 0.05
  for ax in fig.axes[1:]:
    box = ax.get_position()
    box.x0 += offset
    box.x1 += offset
    ax.set_position(box)

  return fig


def plot_reliability_diagrams(
    df_main: pd.DataFrame,
    df_reliability: pd.DataFrame,
    family: str,
    rescaling_method: str = "temperature_scaling",
    dataset_name: str = "imagenet(split='validation[20%:]')",
    gce_prefix=plotting.STD_GCE_PREFIX) -> mpl.figure.Figure:
  """Plots acc/calib and reliability diagrams on clean ImageNet (Figure 1)."""

  df_plot, _ = _get_data(
      df_main, gce_prefix, [family], rescaling_methods=[rescaling_method],
      dataset_name=dataset_name)
  df_models = df_plot.drop_duplicates(subset=["ModelName"])
  df_models = df_models.sort_values(by="model_size")
  model_names = df_models.ModelName.to_list()

  # Set up figure:
  num_cols = max(5, len(model_names))
  width = num_cols * 0.80
  fig = plt.figure(figsize=(width, 1.4))
  spec = fig.add_gridspec(ncols=num_cols, nrows=2, height_ratios=[0.4, 1])

  for i in range(num_cols):
    if i >= len(model_names):
      # Add axes as placeholders for formatting but set to invisible:
      fig.add_subplot(spec[0, i]).set_visible(False)
      fig.add_subplot(spec[1, i]).set_visible(False)
      continue
    model_name = model_names[i]
    # Get predictions:
    mask = df_main.ModelName == model_name
    mask &= df_main.rescaling_method == rescaling_method
    mask &= df_main.Metric == "accuracy"
    mask &= df_main.DatasetName == dataset_name
    raw_model_name = df_main[mask].RawModelName
    assert len(raw_model_name) == 1
    binned = _get_binned_reliability_data(
        df_reliability, utils.assert_and_get_constant(raw_model_name),
        dataset_name)

    rel_ax = fig.add_subplot(spec[1, i])
    _plot_confidence_and_reliability(
        conf_ax=fig.add_subplot(spec[0, i]),
        rel_ax=rel_ax, binned=binned, model_name=model_name)
    if rescaling_method == "none":
      rel_ax.set_xlabel("Confidence\n(unscaled)")
    elif rescaling_method == "temperature_scaling":
      rel_ax.set_xlabel("Confidence\n(temp. scaled)")

  def prettify(s):
    s = display.prettify(s)
    s = s.replace("MLP-Mixer-", "MLP-Mixer\n")
    return s

  plotting.apply_to_fig_text(fig, prettify)
  plotting.apply_to_fig_text(fig, lambda x: x.replace("EfficientNet", "EffNet"))

  fig.subplots_adjust(hspace=-0.05)

  return fig


def _plot_confidence_and_reliability(conf_ax: mpl.axes.Axes,
                                     rel_ax: mpl.axes.Axes,
                                     binned: Dict[str, np.ndarray],
                                     model_name: str,
                                     first_col: int = 0) -> None:
  """Plots a confidence hist and reliability diagram into the provided axes."""

  plot_single_reliability_diagram(rel_ax, binned, zorder=100)

  # Plot and format confidence histogram (top row):
  ax = conf_ax
  bin_widths = np.diff(binned["edges"])
  bin_centers = binned["edges"][:-1] + bin_widths / 2
  ax.bar(
      x=bin_centers,
      height=binned["count"] / np.sum(binned["count"]),
      width=bin_widths,
      color="royalblue",
      edgecolor="k",
      linewidth=0.5,
      zorder=100)
  ax.set_title(model_name, fontsize=6)
  if plotting.col_num(ax) == first_col:
    ax.set_ylabel("Sample frac.", labelpad=2)
    ax.set_yticks([0, 0.5, 1.0])
  else:
    ax.set_yticklabels([])
  ax.set_xticks([])
  ax.set_xlim(0.0, 1.0)
  ax.set_ylim(0.0, 1.0)
  ax.grid(False, which="minor")
  ax.grid(True, axis="y", zorder=-1000)
  ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))

  # Format reliability diaram (bottom row):
  ax = rel_ax
  ax.plot([0, 1], [0, 1], "-", color="gray", linewidth=1, zorder=1000)
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.set_aspect("equal")
  if plotting.col_num(ax) == first_col:
    ax.set_ylabel("Accuracy", labelpad=2)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
  else:
    ax.set_yticklabels([])
  ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
  ax.set_xticklabels([0, "", "", "", "", 1.0])
  ax.grid(False, axis="x")
  ax.grid(False, axis="y", which="minor")
  ax.grid(True, axis="y", which="major")
  ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))


def plot_single_reliability_diagram(ax: mpl.axes.Axes, binned: Dict[str,
                                                                    np.ndarray],
                                    **bar_kwargs):
  """Plots a reliability diagram into ax."""
  bin_widths = np.diff(binned["edges"])
  bin_centers = binned["edges"][:-1] + bin_widths / 2
  ax.bar(
      x=bin_centers,
      height=binned["acc"],
      width=bin_widths,
      color="royalblue",
      edgecolor="k",
      linewidth=0.5,
      **bar_kwargs)

  ax.bar(
      x=bin_centers,
      bottom=binned["acc"],
      height=binned["conf"] - binned["acc"],
      width=bin_widths,
      fc=(1, 0, 0, 0.1),
      edgecolor=(1, 0.3, 0.3),
      linewidth=0.5,
      **bar_kwargs)


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


def _get_binned_reliability_data(
    df_reliability: pd.DataFrame,
    raw_model_name: str,
    dataset_name: str,
    num_bins: int = 10,
    adaptive: bool = False) -> Dict[str, np.ndarray]:
  """Extracts the reliability data for one model and dataset from the df."""
  mask = df_reliability.RawModelName == raw_model_name
  mask &= df_reliability.DatasetName == dataset_name
  mask &= df_reliability.num_bins == num_bins
  mask &= df_reliability.adaptive == adaptive
  num_found = np.sum(mask)
  if not num_found:
    raise ValueError("Found now reliability data for {raw_model_name} "
                     "{dataset_name} num_bins={num_bins} adaptive={adaptive}.")
  elif num_found > 1:
    raise ValueError("Found more than one data row for {raw_model_name} "
                     "{dataset_name} num_bins={num_bins} adaptive={adaptive}.")

  row = df_reliability[mask].iloc[0, :]
  conf = np.array([row[f"conf_{i}"] for i in range(row["num_bins"])])
  acc = np.array([row[f"acc_{i}"] for i in range(row["num_bins"])])
  edges = np.array([row[f"edge_{i}"] for i in range(row["num_bins"])])
  count = np.array([row[f"count_{i}"] for i in range(row["num_bins"])])
  bin_widths = np.diff(edges)
  assert np.all(bin_widths >= 0), f"Edges must be sorted, but are not: {edges}."

  # Ensure top and bottom edges are included:
  eps = np.min(bin_widths) / 2
  if edges[0] > eps:
    edges = np.insert(edges, 0, 0)
  if edges[-1] < (1.0 - eps):
    edges = np.insert(edges, -1, 1.0)

  return {"conf": conf, "acc": acc, "edges": edges, "count": count}
