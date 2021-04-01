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
"""Metrics that are defined only for the synthetic data."""

import collections
import os
from typing import Dict, Text

import numpy as np
import pandas as pd
from robustness_metrics.common import types
from robustness_metrics.metrics import base as metrics_base
import tensorflow as tf


def get_metadata(variant):
  fname = ("https://s3.us-east-1.amazonaws.com/si-score-dataset/"
           f"object_{variant}/metadata.csv")
  return pd.read_csv(fname)




def parse_metadata(variant, fields=None):
  """Returns the filename to group mapping."""
  fields = fields or []
  file_to_group_mapping = {}

  metadata = get_metadata(variant)[["image_id"] + fields]

  # For example "/../area/pidgeon/1.jpg" gets mapped to `area(xxx)` where
  # `xxx` is read from the metadata file.
  for row in metadata.values:
    instance_id = int(row[0])
    if len(row[1:]) == 1:
      group = "(%s)" % row[1]
    elif len(row[1:]) == 2:
      group = "(%.2f,%.2f)" % (float(row[1]), float(row[2]))
    else:
      raise ValueError("Unexpected number of fields: %d" % len(row[:1]))
    file_to_group_mapping[instance_id] = group
  return file_to_group_mapping


@metrics_base.registry.register("synthetic")
class Synthetic(metrics_base.Metric):
  """Synthetic data experiments.

  There are three datasets, and we return the average accuracy for each
  dimension of each dataset:
  (1) Size: we vary the object area in some range (e.g. 10%, 20%, ...), and
      return the average accuracy for each area size.
  (2) Rotation: we vary the object rotation (0 deg, 5 deg, ...), and return
      the average accuracy for each rotation.
  (3) Location: we vary the object offste with respect to the top-left corner
      of the image (e.g. [5%, 5%] and return the average accuracy for each
      offset.
  """

  def __init__(self, dataset_info=None):
    """Synthetic data metric.

    Args:
      dataset_info: DatasetInfo object containing useful information.
    """
    super().__init__(dataset_info)
    self._groups = collections.defaultdict(list)
    self._dataset_info = dataset_info

    self.location_map = parse_metadata("location",
                                       fields=["x_coord", "y_coord"])
    self.size_map = parse_metadata("size", fields=["area"])
    self.rotation_map = parse_metadata("rotation", fields=["rotation"])

  def map_path_to_group(self, image_id, dataset_variant):
    if isinstance(dataset_variant, bytes):
      dataset_variant = dataset_variant.decode("utf-8")

    instance_id = int(image_id)
    if dataset_variant == "location":
      return "location" + self.location_map[instance_id]
    elif dataset_variant == "size":
      return "size" + self.size_map[instance_id]
    elif dataset_variant == "rotation":
      return "rotation" + self.rotation_map[instance_id]
    else:
      raise ValueError("Couldn't map id %s of variant %s" %
                       (image_id, dataset_variant))

  def add_predictions(self,
                      model_predictions: types.ModelPredictions,
                      metadata) -> None:
    for prediction in model_predictions.predictions:
      image_id = metadata["image_id"]
      dataset_variant = metadata["dataset_variant"]
      # Example group IDs: `area(0.5)` or `rotation(121)`.
      group_id = self.map_path_to_group(image_id, dataset_variant)

      correct = metadata["label"]
      predicted = np.argmax(prediction)
      self._groups[group_id].append(int(predicted == correct))

  def result(self) -> Dict[Text, float]:
    scores = {}
    size_scores = []
    rotation_scores = []
    location_scores = []
    for group_id, element_scores in self._groups.items():
      scores[group_id] = np.mean(element_scores)
      if "size" in group_id:
        size_scores.append(scores[group_id])
      elif "rotation" in group_id:
        rotation_scores.append(scores[group_id])
      elif "location" in group_id:
        location_scores.append(scores[group_id])
    scores["size_average"] = np.mean(size_scores)
    scores["rotation_average"] = np.mean(rotation_scores)
    scores["location_average"] = np.mean(location_scores)
    return scores
