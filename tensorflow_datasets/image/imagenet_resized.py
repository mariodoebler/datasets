# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
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

"""Resized imagenet to 8x8, 16x16, 32x32.

This is not to be confused with `downsampled_imagenet` which is a unsupervised
dataset used for generative modeling.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import itertools
import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """@article{chrabaszcz2017downsampled,
  title={A downsampled variant of imagenet as an alternative to the cifar datasets},
  author={Chrabaszcz, Patryk and Loshchilov, Ilya and Hutter, Frank},
  journal={arXiv preprint arXiv:1707.08819},
  year={2017}
}
"""

_DESCRIPTION = """\
This dataset consists of the imagenet dataset resized to {size}x{size}.
The images here are the ones provided by Chrabaszcz et. al. using the box resize method.

For downsampled imagenet for unsupervised learning see `downsampled_imagenet`.
"""

_LABELS_FNAME = 'image/imagenet_resized_labels.txt'
_URL_PREFIX = 'http://www.image-net.org/image/downsample/'


class ImagenetResizedConfig(tfds.core.BuilderConfig):
  """BuilderConfig for Imagenet Resized."""

  def __init__(self, size, **kwargs):
    super(ImagenetResizedConfig, self).__init__(
        version=tfds.core.Version('0.1.0'), **kwargs)
    self.size = size


def _make_builder_configs():
  configs = []
  for size in [8, 16, 32, 64]:
    configs.append(
        ImagenetResizedConfig(
            name='%dx%d' % (size, size),
            size=size,
            description=_DESCRIPTION.format(size=size)))
  return configs


class ImagenetResized(tfds.core.GeneratorBasedBuilder):
  """Imagenet Resized dataset."""

  VERSION = tfds.core.Version('0.1.0')
  BUILDER_CONFIGS = _make_builder_configs()

  def _info(self):
    names_file = tfds.core.get_tfds_path(_LABELS_FNAME)
    size = self.builder_config.size
    return tfds.core.DatasetInfo(
        builder=self,
        description=self.builder_config.description,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(size, size, 3), dtype=np.uint8),
            'label': tfds.features.ClassLabel(names_file=names_file)
        }),
        supervised_keys=('image', 'label'),
        urls=[
            'http://image-net.org/download-images',
            'https://patrykchrabaszcz.github.io/Imagenet32/'
        ],
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    size = self.builder_config.size

    if size in [8, 16, 32]:
      train_path, val_path = dl_manager.download([
          '%s/Imagenet%d_train_npz.zip' % (_URL_PREFIX, size),
          '%s/Imagenet%d_val_npz.zip' % (_URL_PREFIX, size)
      ])
      train_paths = [train_path]
    elif size == 64:
      # 64x64 uses more than one file due to its size.
      train1_path, train2_path, val_path = dl_manager.download([
          '%s/Imagenet64_train_part1_npz.zip' % (_URL_PREFIX),
          '%s/Imagenet64_train_part2_npz.zip' % (_URL_PREFIX),
          '%s/Imagenet64_val_npz.zip' % (_URL_PREFIX)
      ])
      train_paths = [train1_path, train2_path]
    else:
      raise ValueError('Size not implemented!')

    if not all([tf.io.gfile.exists(train_path) for train_path in train_paths
               ]) or not tf.io.gfile.exists(val_path):
      msg = 'You must download the dataset files manually and place them in: '
      msg += ', '.join(train_paths + [val_path])
      raise AssertionError(msg)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            num_shards=10,  # Ignored when using a version with S3 experiment.
            gen_kwargs={
                'archive':
                    itertools.chain(*[
                        dl_manager.iter_archive(train_path)
                        for train_path in train_paths
                    ]),
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            num_shards=1,  # Ignored when using a version with S3 experiment.
            gen_kwargs={
                'archive': dl_manager.iter_archive(val_path),
            },
        ),
    ]

  def _generate_examples(self, archive):
    """Yields examples."""
    for fname, fobj in archive:
      content = fobj.read()
      if content:
        fobj_mem = io.BytesIO(content)
        data = np.load(fobj_mem, allow_pickle=False)
        size = self.builder_config.size
        for i, (image, label) in enumerate(zip(data['data'], data['labels'])):
          # Labels in the original dataset are 1 indexed so we subtract 1 here.
          record = {
              'image': np.reshape(image, (3, size, size)).transpose(1, 2, 0),
              'label': label - 1,
          }
          yield fname + str(i), record
