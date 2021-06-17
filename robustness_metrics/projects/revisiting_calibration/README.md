# Revisiting the Calibration of Modern Neural Networks

by Matthias Minderer, Josip Djolonga, Rob Romijnders, Frances Hubis, Xiaohua Zhai, Neil Houlsby, Dustin Tran, and Mario Lucic.

## Introduction
In this repository we release the code and dataset for the paper [Revisiting the Calibration of Modern Neural Networks](https://arxiv.org/abs/2106.07998). For details on installing the `robustness_metrics` library, see the main repository's [README](https://github.com/google-research/robustness_metrics).

## Dataset

Our dataset of calibration measurements includes 185 distinct models, 79 evaluation datasets, and 28 metric variants. The dataset can be downloaded [here](http://storage.googleapis.com/gresearch/revisiting-calibration/index.html). Check out the [Colab](https://colab.research.google.com/github/google-research/robustness_metrics/blob/master/robustness_metrics/projects/revisiting_calibration/revisiting_calibration.ipynb) for details on how to use the dataset.

## Colab

We provide a Colab that shows how to load the data, how the dataset is organized, and how to reproduce most of the figures from the paper:

https://colab.research.google.com/github/google-research/robustness_metrics/blob/master/robustness_metrics/projects/revisiting_calibration/revisiting_calibration.ipynb

The Colab loads the code from this repository and can be run in a standard CPU runtime.

## Bibtex
```
@article{minderer2021,
  title={Revisiting the Calibration of Modern Neural Networks},
  author={Minderer, Matthias and Djolonga, Josip and Romijnders, Rob and Hubis, Frances and Zhai, Xiaohua and Houlsby, Neil and Tran, Dustin and Lucic, Mario},
  journal={arXiv preprint arXiv:2106.07998},
  year={2021}
}
```

