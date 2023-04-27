# coding=utf-8
# Copyright 2023 The Robustness Metrics Authors.
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

"""Test import."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import robustness_metrics  # pylint: disable=unused-import
import tensorflow as tf


class ImportTest(tf.test.TestCase):

  def test_import(self):
    pass


if __name__ == '__main__':
  tf.test.main()
