import tensorflow as tf


SHUFFLE_BUFFER_SIZE = 10000
NUM_FILES_READ_IN_PARALLEL = 10
NUM_PARALLEL_CALLS = 8
IMAGE_SIZE = 256


class Pipeline:
    def __init__(self, filenames, is_training, batch_size):
        """
        Arguments:
            filenames: a list of strings, paths to tfrecords files.
            is_training: a boolean.
            batch_size, num_epochs: integers.
        """
        self.is_training = is_training

        # read the files in parallel
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        num_shards = len(filenames)
        if is_training:
            dataset = dataset.shuffle(buffer_size=num_shards)
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=NUM_FILES_READ_IN_PARALLEL
        ))
        dataset = dataset.prefetch(buffer_size=batch_size)

        if is_training:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        dataset = dataset.repeat(None if is_training else 1)

        # decode and augment data
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            self.parse_and_preprocess, batch_size=batch_size,
            num_parallel_batches=1, drop_remainder=False
        ))
        dataset = dataset.prefetch(buffer_size=1)

        self.dataset = dataset

    def parse_and_preprocess(self, example_proto):
        """What this function does:
        1. Parses one record from a tfrecords file and decodes it.
        2. Possibly augments it.

        Returns:
            image: a float tensor with shape [height, width, 3],
                a RGB image with pixel values in the range [0, 1].
            label: an int tensor with shape [].
        """
        features = {'image': tf.FixedLenFeature([], tf.string)}
        parsed_features = tf.parse_single_example(example_proto, features)

        # get an image as a string, it will be decoded later
        image_as_string = parsed_features['image']

        if self.is_training:
            image = get_random_crop(image_as_string, crop_size=IMAGE_SIZE)
            image = tf.image.random_flip_left_right(image)
            image = (1.0 / 255.0) * tf.to_float(image)  # to [0, 1] range
            image = random_color_manipulations(image, probability=0.1, grayscale_probability=0.05)
            image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
        else:
            image = tf.image.decode_jpeg(image_as_string, channels=3)
            image = (1.0 / 255.0) * tf.to_float(image)  # to [0, 1] range

        features = image
        return features


def get_random_crop(image_as_string, crop_size):

    crop_size = tf.constant(crop_size, tf.int32)
    shape = tf.image.extract_jpeg_shape(image_as_string)
    h, w = shape[0], shape[1]
    # it assumed that min(h, w) > crop_size

    y = tf.random_uniform([], 0, h - crop_size, dtype=tf.int32)
    x = tf.random_uniform([], 0, w - crop_size, dtype=tf.int32)
    crop_window = tf.stack([y, x, crop_size, crop_size])

    crop = tf.image.decode_and_crop_jpeg(image_as_string, crop_window, channels=3)
    return crop


def random_color_manipulations(image, probability=0.1, grayscale_probability=0.1):

    def manipulate(image):
        br_delta = tf.random_uniform([], -32.0/255.0, 32.0/255.0)
        cb_factor = tf.random_uniform([], -0.1, 0.1)
        cr_factor = tf.random_uniform([], -0.1, 0.1)
        channels = tf.split(axis=2, num_or_size_splits=3, value=image)
        red_offset = 1.402 * cr_factor + br_delta
        green_offset = -0.344136 * cb_factor - 0.714136 * cr_factor + br_delta
        blue_offset = 1.772 * cb_factor + br_delta
        channels[0] += red_offset
        channels[1] += green_offset
        channels[2] += blue_offset
        image = tf.concat(axis=2, values=channels)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def to_grayscale(image):
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)
        return image

    with tf.name_scope('random_color_manipulations'):
        do_it = tf.less(tf.random_uniform([]), probability)
        image = tf.cond(do_it, lambda: manipulate(image), lambda: image)

    with tf.name_scope('to_grayscale'):
        do_it = tf.less(tf.random_uniform([]), grayscale_probability)
        image = tf.cond(do_it, lambda: to_grayscale(image), lambda: image)

    return image
