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
"""Implementation of data preprocessing ops.

All preprocessing ops should return a data processing functors. A data
is represented as a dictionary of tensors, where field "image" is reserved
for 3D images (height x width x channels). The functors output dictionary with
field "image" being modified. Potentially, other fields can also be modified
or added.
"""
import abc
import collections
from robustness_metrics.common import registry
import tensorflow as tf2
import tensorflow.compat.v1 as tf


class PreprocessingOp(metaclass=abc.ABCMeta):
  """The abstract class representing a preprocessing operation."""


registry = registry.Registry(PreprocessingOp)
get = registry.get


def tf_apply_to_image_or_images(fn, image_or_images, **map_kw):
  """Applies a function to a single image or each image in a batch of them.

  Args:
    fn: the function to apply, receives an image, returns an image.
    image_or_images: Either a single image, or a batch of images.
    **map_kw: Arguments passed through to tf.map_fn if called.

  Returns:
    The result of applying the function to the image or batch of images.

  Raises:
    ValueError: if the input is not of rank 3 or 4.
  """
  static_rank = len(image_or_images.get_shape().as_list())
  if static_rank == 3:  # A single image: HWC
    return fn(image_or_images)
  elif static_rank == 4:  # A batch of images: BHWC
    return tf.map_fn(fn, image_or_images, **map_kw)
  elif static_rank > 4:  # A batch of images: ...HWC
    input_shape = tf.shape(image_or_images)
    h, w, c = image_or_images.get_shape().as_list()[-3:]
    image_or_images = tf.reshape(image_or_images, [-1, h, w, c])
    image_or_images = tf.map_fn(fn, image_or_images, **map_kw)
    return tf.reshape(image_or_images, input_shape)
  else:
    raise ValueError("Unsupported image rank: %d" % static_rank)


class BatchedPreprocessing(object):
  """Decorator for preprocessing ops, which adds support for image batches."""

  def __init__(self, output_dtype=None, data_key="image"):
    self.output_dtype = output_dtype
    self.data_key = data_key

  def __call__(self, get_pp_fn):

    def get_batch_pp_fn(*args, **kwargs):
      """Preprocessing function that supports batched images."""

      def pp_fn(image):
        return get_pp_fn(*args, **kwargs)({self.data_key: image})[self.data_key]

      def _batch_pp_fn(data):
        image = data[self.data_key]
        data[self.data_key] = tf_apply_to_image_or_images(
            pp_fn, image, dtype=self.output_dtype)
        return data
      return _batch_pp_fn
    return get_batch_pp_fn


def maybe_repeat(arg, n_reps):
  if not isinstance(arg, collections.Sequence):
    arg = (arg,) * n_reps
  return arg


@registry.register("color_distort")
class ColorDistort(PreprocessingOp):
  """Applies random brigthness/saturation/hue/contrast transformations."""

  @staticmethod
  @BatchedPreprocessing()
  def apply():
    """Applies random brigthness/saturation/hue/contrast transformations."""
    def _color_distortion(data):
      image = data["image"]
      image = tf.image.random_brightness(image, max_delta=128. / 255.)
      image = tf.image.random_saturation(image, lower=0.1, upper=2.0)
      image = tf.image.random_hue(image, max_delta=0.5)
      image = tf.image.random_contrast(image, lower=0.1, upper=2.0)
      data["image"] = image
      return data
    return _color_distortion


@registry.register("decode_unicode")
class DecodeUnicode(PreprocessingOp):
  """Converts unicode to int array."""

  @staticmethod
  def apply(key, fixed_length=256):
    """Converts unicode to int array.

    This function is useful to pass unicode through TPUs, which supports
    currently only int/float types.

    Args:
      key: key of the unicode field in the input data dict.
      fixed_length: TPU requires fixed shape arrays. The int array will be
        padded to fixed_length, with all zeros.

    Returns:
      A function that decodes the unicode value to a fixed length list.
    """

    def _dynamic_padding(inp, min_size):
      """Padding an input vector to min_size."""
      pad_size = min_size - tf.shape(inp)[0]
      paddings = [[0, pad_size]]
      return tf.pad(inp, paddings)

    def _decode_unicode(data):
      """Decode unicode to int array."""
      if key in data:
        decode = tf.strings.unicode_decode(data[key], "UTF-8")
        decode = _dynamic_padding(decode, fixed_length)
        decode.set_shape(fixed_length)
        data[key] = decode
      else:
        tf.logging.error(
            "Key {} not found from {}.".format(key, data), exc_info=True)
      return data

    return _decode_unicode


@registry.register("random_brightness")
class RandomBrightness(PreprocessingOp):
  """Adds a random small value to all pixel intensities."""

  @staticmethod
  @BatchedPreprocessing()
  def apply(max_delta=0.1):
    """Applies random brigthness transformations."""

    # A random value in [-max_delta, +max_delta] is added to the image values.
    # Small max_delta <1.0 assumes that the image values are within [0, 1].
    def _random_brightness(data):
      image = data["image"]
      image = tf.image.random_brightness(image, max_delta)
      data["image"] = image
      return data

    return _random_brightness


@registry.register("random_saturation")
class RandomSaturation(PreprocessingOp):
  """Applies random saturation transformations."""

  @staticmethod
  @BatchedPreprocessing()
  def apply(lower=0.5, upper=2.0):
    """Applies random saturation transformations."""

    def _random_saturation(data):
      # Multiplies saturation channel in HSV (with converting from/to RGB) with
      # a random float value in [lower, upper].
      image = data["image"]
      image = tf.image.random_saturation(image, lower=lower, upper=upper)
      data["image"] = image
      return data
    return _random_saturation


@registry.register("random_hue")
class RandomHue(PreprocessingOp):
  """Adds a random offset to hue channel in HSV."""

  @staticmethod
  @BatchedPreprocessing()
  def get_random_hue(max_delta=0.1):
    """Applies random hue transformations."""

    def _random_hue(data):
      # Adds to hue channel in HSV (with converting from/to RGB) a random offset
      # in [-max_delta, +max_delta].
      image = data["image"]
      image = tf.image.random_hue(image, max_delta=max_delta)
      data["image"] = image
      return data
    return _random_hue


@registry.register("random_contrast")
class RandomContrast(PreprocessingOp):
  """Applies a random contrast change."""

  @staticmethod
  @BatchedPreprocessing()
  def apply(lower=0.5, upper=2.0):
    """Applies random contrast transformations."""

    def _random_contrast(data):
      # Stretches/shrinks value stddev (per channel) by multiplying with a
      # random value in [lower, upper].
      image = data["image"]
      image = tf.image.random_contrast(image, lower=lower, upper=upper)
      data["image"] = image
      return data
    return _random_contrast


@registry.register("drop_channels")
class DropChannels(PreprocessingOp):
  """Drops 2 out of 3 channels  ."""

  @staticmethod
  @BatchedPreprocessing()
  def apply(keep_original=0.25, noise_min=-1.0, noise_max=1.0):
    """Drops 2/3 channels and fills the remaining channels with noise."""

    def _drop_channels(data):
      image = data["image"]

      def _drop(keep_i):
        shape = image.get_shape().as_list()
        size, num_channels = shape[:-1], shape[-1]
        return tf.concat([
            image[:, :, i:i + 1] if i == keep_i else tf.random_uniform(
                size + [1], noise_min, noise_max) for i in range(num_channels)
        ],
                         axis=2)

      def _drop_random_channel(coin_channel):
        return tf.case({
            tf.equal(coin_channel, 0): lambda: _drop(0),
            tf.equal(coin_channel, 1): lambda: _drop(1),
            tf.equal(coin_channel, 2): lambda: _drop(2),
        })

      coin_keep_original = tf.random.uniform([], 0.0, 1.0, dtype=tf.float32)
      coin_channel = tf.random.uniform([], 0, 3, dtype=tf.int32)
      image = tf.case({
          tf.less(coin_keep_original, keep_original):
              lambda: image,
          tf.greater_equal(coin_keep_original, keep_original):
              lambda: _drop_random_channel(coin_channel)
      })
      data["image"] = image
      return data

    return _drop_channels


@registry.register("decode")
class DecodeImage(PreprocessingOp):
  """Decode an encoded image string, see tf.io.decode_image."""

  @staticmethod
  def apply(key="image", channels=3):
    """Decode an encoded image string, see tf.io.decode_image."""
    def _decode(data):
      # tf.io.decode_image does not set the shape correctly, so we use
      # tf.io.deocde_jpeg, which also works for png, see
      # https://github.com/tensorflow/tensorflow/issues/8551
      data[key] = tf.io.decode_jpeg(data[key], channels=channels)
      return data
    return _decode


@registry.register("pad")
class Pad(PreprocessingOp):
  """Pads an image."""

  @staticmethod
  @BatchedPreprocessing()
  def apply(pad_size):
    """Pads an image.

    Args:
      pad_size: either an integer u giving verticle and horizontal pad sizes u,
        or a list or tuple [u, v] of integers where u and v are vertical and
        horizontal pad sizes.

    Returns:
      A function for padding an image.
    """
    pad_size = maybe_repeat(pad_size, 2)

    def _pad(data):
      image = data["image"]
      image = tf.pad(
          image,
          [[pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [0, 0]])
      data["image"] = image
      return data

    return _pad


@registry.register("resize")
class Resize(PreprocessingOp):
  """Resizes image to a given size."""

  @staticmethod
  @BatchedPreprocessing()
  def apply(resize_size):
    """Resizes image to a given size.

    Args:
      resize_size: either an integer H, where H is both the new height and width
        of the resized image, or a list or tuple [H, W] of integers, where H and
        W are new image"s height and width respectively.

    Returns:
      A function for resizing an image.
    """
    resize_size = maybe_repeat(resize_size, 2)

    def _resize(data):
      """Resizes image to a given size."""
      image = data["image"]
      # Note: use TF-2 version of tf.image.resize as the version in TF-1 is
      # buggy: https://github.com/tensorflow/tensorflow/issues/6720.
      dtype = image.dtype
      image = tf2.image.resize(image, resize_size)
      image = tf.cast(image, dtype)
      data["image"] = image
      return data

    return _resize


@registry.register("resize_small")
class ResizeSmall(PreprocessingOp):
  """Resizes the smaller side to a desired value keeping the aspect ratio."""

  @staticmethod
  @BatchedPreprocessing()
  def apply(smaller_size):
    """Resizes the smaller side to `smaller_size` keeping aspect ratio.

    Args:
      smaller_size: an integer, that represents a new size of the smaller side
        of an input image.
    Returns:
      A function, that resizes an image and preserves its aspect ratio.
    """

    def _resize_small(data):
      image = data["image"]
      h, w = tf.shape(image)[0], tf.shape(image)[1]
      ratio = (
          tf.cast(smaller_size, tf.float32) /
          tf.cast(tf.minimum(h, w), tf.float32))
      h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
      w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)
      data["image"] = tf.image.resize_area(image[None], [h, w])[0]
      return data

    return _resize_small


@registry.register("inception_crop")
class InceptionCrop(PreprocessingOp):
  """Applies an Inception-style image crop."""

  @staticmethod
  @BatchedPreprocessing()
  def apply(resize_size=None, area_min=5, area_max=100):
    """Applies an Inception-style image crop.

    Inception-style crop is a random image crop (its size and aspect ratio are
    random) that was used for training Inception models, see
    https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf.

    Args:
      resize_size: Resize image to [resize_size, resize_size] after crop.
      area_min: minimal crop area.
      area_max: maximal crop area.

    Returns:
      A function, that applies inception crop.
    """

    def _inception_crop(data):  # pylint: disable=missing-docstring
      image = data["image"]
      begin, size, _ = tf.image.sample_distorted_bounding_box(
          tf.shape(image),
          tf.zeros([0, 0, 4], tf.float32),
          area_range=(area_min / 100, area_max / 100),
          min_object_covered=0,  # Don't enforce a minimum area.
          use_image_if_no_bounding_boxes=True)
      data["image"] = tf.slice(image, begin, size)
      # Unfortunately, the above operation loses the depth-dimension. So we need
      # to restore it the manual way.
      data["image"].set_shape([None, None, image.shape[-1]])
      if resize_size:
        data["image"] = Resize.apply([resize_size, resize_size])(data)["image"]
      return data

    return _inception_crop


@registry.register("decode_jpeg_and_inception_crop")
class DecodeAndInceptionCrop(PreprocessingOp):
  """Decode jpeg string and make inception-style image crop."""

  @staticmethod
  def apply(resize_size=None, area_min=5, area_max=100):
    """Decode jpeg string and make inception-style image crop.

    Inception-style crop is a random image crop (its size and aspect ratio are
    random) that was used for training Inception models, see
    https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf.

    Args:
      resize_size: Resize image to [resize_size, resize_size] after crop.
      area_min: minimal crop area.
      area_max: maximal crop area.

    Returns:
      A function, that applies inception crop.
    """

    def _inception_crop(data):  # pylint: disable=missing-docstring
      image = data["image"]
      shape = tf.image.extract_jpeg_shape(image)
      begin, size, _ = tf.image.sample_distorted_bounding_box(
          shape,
          tf.zeros([0, 0, 4], tf.float32),
          area_range=(area_min / 100, area_max / 100),
          min_object_covered=0,  # Don't enforce a minimum area.
          use_image_if_no_bounding_boxes=True)
      # Crop the image to the specified bounding box.
      offset_y, offset_x, _ = tf.unstack(begin)
      target_height, target_width, _ = tf.unstack(size)
      crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
      image = tf.image.decode_and_crop_jpeg(image, crop_window, channels=3)
      data["image"] = image

      if resize_size:
        data["image"] = Resize.apply([resize_size, resize_size])(data)["image"]
      return data

    return _inception_crop


@registry.register("random_crop")
class RandomCrop(PreprocessingOp):
  """Makes a random crop of a given size."""

  @staticmethod
  @BatchedPreprocessing()
  def apply(crop_size):
    """Makes a random crop of a given size.

    Args:
      crop_size: either an integer H, where H is both the height and width of
        the random crop, or a list or tuple [H, W] of integers, where H and W
        are height and width of the random crop respectively.

    Returns:
      A function, that applies random crop.
    """
    crop_size = maybe_repeat(crop_size, 2)

    def _crop(data):
      image = data["image"]
      h, w, c = crop_size[0], crop_size[1], image.shape[-1]
      image = tf.random_crop(image, [h, w, c])
      data["image"] = image
      return data

    return _crop


@registry.register("central_crop")
class CentralCrop(PreprocessingOp):
  """Flips an image horizontally with probability 50%."""

  @staticmethod
  @BatchedPreprocessing()
  def apply(crop_size):
    """Makes central crop of a given size.

    Args:
      crop_size: either an integer H, where H is both the height and width of
        the central crop, or a list or tuple [H, W] of integers, where H and W
        are height and width of the central crop respectively.

    Returns:
      A function, that applies central crop.
    """
    crop_size = maybe_repeat(crop_size, 2)

    def _crop(data):
      image = data["image"]
      h, w = crop_size[0], crop_size[1]
      dy = (tf.shape(image)[0] - h) // 2
      dx = (tf.shape(image)[1] - w) // 2
      image = tf.image.crop_to_bounding_box(image, dy, dx, h, w)
      data["image"] = image
      return data

    return _crop


@registry.register("flip_lr")
class FlipLeftRight(PreprocessingOp):
  """Flips an image horizontally with probability 50%."""

  @staticmethod
  @BatchedPreprocessing()
  def apply():
    """Flips an image horizontally with probability 50%."""
    def _random_flip_lr_pp(data):
      image = data["image"]
      image = tf.image.random_flip_left_right(image)
      data["image"] = image
      return data

    return _random_flip_lr_pp


@registry.register("flip_ud")
class FlipUpDown(PreprocessingOp):
  """Flips an image vertically with probability 50%."""

  @staticmethod
  @BatchedPreprocessing()
  def apply():
    """Flips an image vertically with probability 50%."""
    def _random_flip_ud_pp(data):
      image = data["image"]
      image = tf.image.random_flip_up_down(image)
      data["image"] = image
      return data

    return _random_flip_ud_pp


@registry.register("random_rotate90")
class RandomRotate90(PreprocessingOp):
  """Randomly rotate an image by multiples of 90 degrees."""

  @staticmethod
  @BatchedPreprocessing()
  def apply():
    """Randomly rotate an image by multiples of 90 degrees."""
    def _random_rotation90(data):
      """Rotation function."""
      image = data["image"]
      num_rotations = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
      image = tf.image.rot90(image, k=num_rotations)
      data["image"] = image
      return data

    return _random_rotation90


@registry.register("value_range")
class ValueRange(PreprocessingOp):
  """Transforms a [in_min,in_max] image to [vmin,vmax] range."""

  @staticmethod
  @BatchedPreprocessing(output_dtype=tf.float32)
  def apply(vmin=-1, vmax=1, in_min=0, in_max=255.0, clip_values=False):
    """Transforms a [in_min,in_max] image to [vmin,vmax] range.

    Input ranges in_min/in_max can be equal-size lists to rescale the invidudal
    channels independently.

    Args:
      vmin: A scalar. Output max value.
      vmax: A scalar. Output min value.
      in_min: A scalar or a list of input min values to scale. If a list, the
        length should match to the number of channels in the image.
      in_max: A scalar or a list of input max values to scale. If a list, the
        length should match to the number of channels in the image.
      clip_values: Whether to clip the output values to the provided ranges.

    Returns:
      A function to rescale the values.
    """

    def _value_range(data):
      """Scales values in given range."""
      in_min_t = tf.constant(in_min, tf.float32)
      in_max_t = tf.constant(in_max, tf.float32)
      image = tf.cast(data["image"], tf.float32)
      image = (image - in_min_t) / (in_max_t - in_min_t)
      image = vmin + image * (vmax - vmin)
      if clip_values:
        image = tf.clip_by_value(image, vmin, vmax)
      data["image"] = image
      return data

    return _value_range


@registry.register("value_range_mc")
class ValueRangeMultichannel(PreprocessingOp):
  """Independent multi-channel rescaling."""

  @staticmethod
  def apply(vmin, vmax, *args):
    """Independent multi-channel rescaling."""
    if len(args) % 2:
      raise ValueError("Additional args must be list of even length giving "
                       "`in_max` and `in_min` concatenated")
    num_channels = len(args) // 2
    in_min = args[:num_channels]
    in_max = args[-num_channels:]
    return ValueRange.apply(vmin, vmax, in_min, in_max)


@registry.register("replicate")
class Replicate(PreprocessingOp):
  """Replicates an image along a new batch dimension."""

  @staticmethod
  def apply(num_replicas=2):
    """Replicates an image `num_replicas` times along a new batch dimension."""
    def _replicate(data):
      tiles = [num_replicas] + [1] * len(data["image"].shape)
      data["image"] = tf.tile(data["image"][None], tiles)
      return data
    return _replicate


@registry.register("standardize")
class Standardize(PreprocessingOp):
  """Standardize an image."""

  @staticmethod
  @BatchedPreprocessing(output_dtype=tf.float32)
  def apply(mean, std):
    """Standardize an image with the given mean and standard deviation."""
    def _standardize(data):
      image = tf.cast(data["image"], dtype=tf.float32)
      data["image"] = (image - mean) / std
      return data
    return _standardize


@registry.register("select_channels")
class SelectChannels(PreprocessingOp):
  """Returns function to select specified channels."""

  @staticmethod
  @BatchedPreprocessing()
  def apply(channels):
    """Returns function to select specified channels."""
    def _select_channels(data):
      """Returns a subset of available channels."""
      data["image"] = tf.gather(data["image"], channels, axis=-1)
      return data
    return _select_channels


@registry.register("onehot")
class OneHotEncoding(PreprocessingOp):
  """One-hot encoding of the input."""

  @staticmethod
  def apply(depth, key="labels", key_result=None, multi=True):
    """One-hot encodes the input.

    Args:
      depth: Length of the one-hot vector (how many classes).
      key: Key of the data to be one-hot encoded.
      key_result: Key under which to store the result (same as `key` if None).
      multi: If there are multiple labels, whether to merge them into the same
        "multi-hot" vector (True) or keep them as an extra dimension (False).
    Returns:
      Data dictionary.
    """

    def _onehot(data):
      onehot = tf.one_hot(data[key], depth)
      if multi and len(onehot.shape) == 2:
        onehot = tf.reduce_max(onehot, axis=0)
      data[key_result or key] = onehot
      return data

    return _onehot


def fingerprint_int64(batch):
  """Returns an tf.int64 hash for each element of the input."""
  hash_bytes = tf.squeeze(tf.fingerprint([batch]))
  # Fingerprint op writes fingerprint values as byte arrays. For example, the
  # default method farmhash64 generates a 64-bit fingerprint value at a time.
  # This 8-byte value is written out as an tf.uint8 array of size 8,
  # in little-endian order. These are then combined in base 8 to get one int64.
  hash_base = tf.constant([[256**i for i in range(8)]], dtype=tf.int64)
  hash_bytes = tf.cast(hash_bytes, dtype=tf.int64)
  element_hashes_int64 = tf.reduce_sum(hash_bytes * hash_base, axis=1)
  return element_hashes_int64


def combine_fingerprints(hashes1, hashes2):
  """Combines two tensors of fingerprints.

  The two tensors have to be compatible (broadcastable) for addition.

  The fingerprints are combined so that the output distribution is roughly
  uniform (assuming that the original hashes are also uniformly distributed).

  Args:
    hashes1: 1-D tensor, tf.int64.
    hashes2: 1-D tensor, tf.int64.

  Returns:
    A 1-D tensor with the hash values combined.
  """
  # Based on Boost combine_hash function, extended to 64 bits. Original code
  # (in 32 bits): hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2)).
  magic_number = -7046029254386353131  # i.e. 0x9E3779B97F4A7C15
  return tf.bitwise.bitwise_xor(
      hashes1, hashes2 + magic_number + tf.bitwise.left_shift(hashes1, 6) +
      tf.bitwise.right_shift(hashes1, 2))


def to_hash_bucket_deterministic(batch, num_buckets, seed=None):
  """Buckets input examples, roughly uniformly and deterministically.

  Args:
    batch: a tensor of rank >= 1, containing the input examples
      (batch axis = 0).
    num_buckets: an integer, number of buckets.
    seed: (optional) this seed will be used in the hash computation so that one
      can obtain pseudo-random but deterministic bucket assignments.

  Returns:
    A tensor of rank 1, containing the bucket assigned to each input example.
  """
  # Note: In order to get deterministic bucketing, the hash function has to be
  # deterministic. That's why we use fingerprint_int64.
  hashes = fingerprint_int64(batch)
  if seed is not None:
    hashes = combine_fingerprints(hashes, fingerprint_int64([seed]))
  return tf.math.mod(hashes, num_buckets)


def compose(*functions):
  """Composes an arbitrary number of functions.

  Assumes that None == Identity function.

  Args:
    *functions: Arbitrary number of callables.

  Returns:
    Composition of said callables.
  """
  def _composed_fn(*x):
    for fn in functions:
      if fn:
        # Note that we cannot use `collections.abc.Iterable` because this will
        # include a `dict` which will be incorrectly passed if not wrapped in a
        # tuple.
        if not isinstance(x, (list, tuple)):
          x = (x,)
        x = fn(*x)
    return x
  return _composed_fn
