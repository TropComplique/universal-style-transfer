import os
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
import math
import io
from PIL import Image


"""
This script creates training and validation data.

Just run:
python create_tfrecords.py

And don't forget set the right paths below.
"""


# paths to downloaded data
IMAGES_DIR = '/home/dan/datasets/COCO/images/'
# (it contains folders train2017 and val2017)

# path where converted data will be stored
RESULT_PATH = '/home/dan/datasets/COCO/ust/'

# because dataset is big we will split it into parts
NUM_TRAIN_SHARDS = 300
NUM_VAL_SHARDS = 1

# images will be resized
MIN_DIMENSION = 512


def to_tf_example(image_path):
    """
    Arguments:
        image_path: a string.
    Returns:
        an instance of tf.train.Example.
    """

    with tf.gfile.GFile(image_path, 'rb') as f:
        encoded_jpg = f.read()

    # check image format
    image = Image.open(io.BytesIO(encoded_jpg))
    if not image.format == 'JPEG':
        return None

    width, height = image.size
    if image.mode == 'L':  # if grayscale
        rgb_image = np.stack(3*[np.array(image)], axis=2)
        encoded_jpg = to_jpeg_bytes(rgb_image)
        image = Image.open(io.BytesIO(encoded_jpg))
    assert image.mode == 'RGB'
    assert width > 0 and height > 0

    original_min_dim = min(height, width)
    scale_factor = MIN_DIMENSION / original_min_dim
    height = int(np.round(height * scale_factor))
    width = int(np.round(width * scale_factor))
    image = image.resize((width, height), Image.LANCZOS)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(to_jpeg_bytes(image)),
    }))
    return example


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def to_jpeg_bytes(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format='jpeg')
    return buffer.getvalue()


def convert(image_dir, result_path, num_shards):

    examples_list = os.listdir(image_dir)
    examples_list = [os.path.join(image_dir, n) for n in examples_list]

    shutil.rmtree(result_path, ignore_errors=True)
    os.mkdir(result_path)

    # randomize image order
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    print('Number of images:', num_examples)

    shard_size = math.ceil(num_examples/num_shards)
    print('Number of images per shard:', shard_size)

    shard_id = 0
    num_examples_written = 0
    num_skipped_images = 0
    for image_path in tqdm(examples_list):

        if num_examples_written == 0:
            shard_path = os.path.join(result_path, 'shard-%04d.tfrecords' % shard_id)
            if not os.path.exists(shard_path):
                writer = tf.python_io.TFRecordWriter(shard_path)

        tf_example = to_tf_example(image_path)
        if tf_example is None:
            num_skipped_images += 1
            continue
        writer.write(tf_example.SerializeToString())
        num_examples_written += 1

        if num_examples_written == shard_size:
            shard_id += 1
            num_examples_written = 0
            writer.close()

    if num_examples_written != 0:
        writer.close()

    print('Number of skipped images:', num_skipped_images)
    print('Number of shards:', shard_id + 1)
    print('Result is here:', result_path, '\n')


shutil.rmtree(RESULT_PATH, ignore_errors=True)
os.mkdir(RESULT_PATH)

image_dir = os.path.join(IMAGES_DIR, 'train2017')
result_path = os.path.join(RESULT_PATH, 'train')
convert(image_dir, result_path, NUM_TRAIN_SHARDS)

image_dir = os.path.join(IMAGES_DIR, 'val2017')
result_path = os.path.join(RESULT_PATH, 'val')
convert(image_dir, result_path, NUM_VAL_SHARDS)
