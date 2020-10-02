## Robustness Metrics

Robustness Metrics provides lightweight modules in order to evaluate the robustness of classification models across three sets of metrics:

1.  out-of-distribution generalization (e.g. a non-expert human would be able to classify similar objects, but possibly changed viewpoint, scene setting or clutter).
2.  stability (of the prediction and predicted probabilities) under natural perturbation of the input.
3.  uncertainty (e.g. assessing to which extent the probabilities predicted by a model reflect the true probabilities)

The library includes popular out-of-distribution datasets (ImageNetV2, ImageNet-C, etc.) and can be readily applied to benchmark arbitrary models and is not limited to vision models: any mapping from input -> logits will do.

## Getting Started

First, install the library and its dependencies as

```sh
python setup.py install
```

or directly from the repository as

```sh
pip install "git+https://github.com/google-research/robustness_metrics.git#egg=robustness_metrics"
```

There are three steps to evaluate a model: 1. import the model; 2. launch an
experiment; and 3. examine results.

### 1. Import the model

Import your model by writing a file that specifies how to make predictions and
how we should pre-process the data. This file contains a single function
`create` that returns a tuple of:

 * a function `predict(features: Dict[str, Tensor]` that takes a batch from the
   dataset and computes your model predictions; and
 * a pre-processing function that will be applied to the dataset.

The latter can be omitted (return `None`) in which case we will use default
pre-processing. For ImageNet, this is a central crop to (224, 224) and scales
the pixel values to the range [-1, +1]. The dictionary holding the inputs
`features` follows the same naming convention as `tensorflow_datasets`. As all
imported datasets are currently image datasets, this means that a batch of
images will be stored in the field `features["image"]`.

<!-- TODO(trandustin): Add GitHub links to filenames displayed in open-source.-->
For examples, see `models`. You don't need to store the model file there.

__Parameterized models.__ Sometimes you have multiple model variants that you
would like to to test, e.g., different model sizes or training datasets. To
achieve this, add arguments to the `create` function, e.g. `create(network_type,
network_width)` to import networks of varying widths and sizes.

__Non-TensorFlow models.__ If your model is not written in TensorFlow, you can
convert the data to numpy and feed those to your model. For example, take a look
at `models/random_imagenet_numpy.py`. Note that we do not provide TPU
support in this case.

### 2. Launch an experiment

You can either run the launcher to compute a specific set of measurements (e.g.
accuracy on ImageNet, expected calibration error on ImageNet-A) which is done
via the `--measurement` flag, or you can compute all the measurements that are
necessary for a specific robustness report, done using the `--report` flag.

Note that the library is using `tensorflow_datasets` to load the data. If
you are loading them for the first time on your system, then it will first
download and serialize them to a local directory.

Launch `bin/compute_report.py`, passing in your model
file in `model_path`. If your `create` function has parameters, you can
pass them via the `--model_args` flag (as Python code, it will be
`literal_eval`'ed).

You can explicitly specify the set of measurements you want to make

```sh
python3 bin/compute_report.py \
   --model_path=models/random_imagenet_numpy.py \
   --measurement="accuracy@imagenet" \
   --measurement="nll@imagenet_v2(variant='MATCHED_FREQUENCY')" \
   --measurement="ece@imagenet_a"
```

or, alternatively, you can use one of the reports we provide, e.g.

```sh
python3 bin/compute_report.py \
   --model_path=models/random_imagenet_numpy.py \
   --report="classification_report(datasets=['imagenet'])"
```

For the list of reports, please see `reports/`.


We provide several models in the directory `models/`, that you can run to
reproduce their results. The models are serialized as `tensorflow_hub`
models and will be automatically downloaded to your disk. For example:

```sh
python3 bin/compute_report.py \
   --model_path=models/bit.py \
   --model_args="dataset='Imagenet21k',network='R50',size='x1'" \
   --measurement="accuracy@imagenet" \
   --measurement="nll@imagenet_v2(variant='MATCHED_FREQUENCY')" \
   --measurement="ece@imagenet_a"
```

### 3. Examine results

To see results, look at the printed output.

