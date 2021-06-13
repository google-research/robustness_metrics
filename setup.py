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

"""Robustness Metrics."""

from setuptools import find_packages
from setuptools import setup

setup(
    name="robustness_metrics",
    version="0.0.1",
    description="Robustness Metrics",
    author="Robustness Metrics Team",
    author_email="noauthor@google.com",
    url="https://github.com/google-research/robustness_metrics",
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=[
        "absl-py",
        "matplotlib>=3.3",
        "pandas",
        "scikit-learn",
        "seaborn",
        "tabulate",
        "tensorflow_datasets",
        "tensorflow_hub",
        "tf-nightly",
        "tfp-nightly",
    ],
    extras_require={},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="robustness uncertainty distribution shift machine learning",
)
