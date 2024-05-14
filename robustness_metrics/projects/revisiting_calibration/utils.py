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

"""Utilities."""

import re
from typing import List

from absl import logging
import numpy as np
import pandas as pd


def re_with_default(pattern, string, default=None):
  """Finds a regex match and returns a default value of no match is found."""
  match = re.search(pattern, string)
  if match:
    return match.group()
  else:
    return default


def assert_and_get_constant(x: pd.Series):
  """Asserts that a series has only one value, and returns it."""
  assert x.nunique() == 1, f"More than one unique value in series: {x}"
  return x.iat[0]


def assert_no_duplicates_in_condition(df: pd.DataFrame, group_by: List[str]):
  """Checks that a DF contains a single row per condition, no duplicates."""
  count = df.groupby(by=group_by).count()
  if count.max().max() <= 1:
    return
  # There are duplicates:
  vals = count.max(axis="columns").idxmax()
  mask = np.ones(df.shape[0], bool)
  for col, val in zip(group_by, vals):
    mask &= df[col] == val
  msg = [f"{col}: {val}" for col, val in zip(group_by, vals)]
  logging.info("\n\nShowing duplicates for %s\n", msg)
  df_dup = df[mask].copy()
  df_dup = df_dup.loc[:, df_dup.apply(pd.Series.nunique) > 1]
  raise ValueError(
      "There are duplicates in some conditions. Check DFs displayed above.\n\n"
      + str(df_dup))


def average_imagenet_c_corruption_types(df: pd.DataFrame,
                                        group_by: List[str]) -> pd.DataFrame:
  """Averages over ImageNet-C corruption types.

  This function groups the rows in `df` by the DatasetName and any additional
  columns provided in `group_by`. It returns a new DataFrame containing the mean
  within each group for numerical columns. For non-numerical columns, the first
  element of the group is used.

  Args:
    df: Metrics DataFrame containing ImageNet-C results.
    group_by: List of additional column names to group by before averaging.

  Returns:
    A copy of `df` in which metrics are averaged within each group.
  """
  df = df.copy()
  if not df.size:  # Empty dataframe.
    logging.warning(
        "Empty DataFrame passed to average_imagenet_c_corruption_types")
    return df
  assert_no_duplicates_in_condition(df, group_by=group_by + ["DatasetName"])
  df.DatasetName = df.DatasetName.map(
      lambda name: "imagenet_c" if name.startswith("imagenet_c") else name)
  grouped = df.groupby(by=group_by + ["DatasetName"], as_index=False)
  # Average over ImageNet-C datasets:
  def mean(x):
    if np.all(np.issubdtype(x.dtype, np.number)):
      return np.mean(x)
    else:
      return x.iloc[0]
  return grouped.aggregate(mean)
