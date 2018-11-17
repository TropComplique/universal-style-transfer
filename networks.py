import tensorflow as tf
import tensorflow.contrib.slim as slim


def encoder(images):
    """
    It is based on classical VGG-19 architecture.

    It implemented in a way that works with
    the official tensorflow pretrained checkpoint
    (https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).

    Arguments:
        images: a float tensor with shape [batch_size, image_height, image_width, 3],
            it represents RGB images with pixel values in range [0, 255].
    Returns:
        a dict with float tensors, they represent all Relu_X_1 features.
    """
    features = {}

    with tf.name_scope('standardize'):
        channel_means = tf.constant([123.15163, 115.902885, 103.06262], dtype=tf.float32)
        x = images - channel_means

    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d], trainable=False, stride=1, padding='VALID'):
            with slim.arg_scope([slim.max_pool2d], stride=2, padding='SAME'):

                x = conv(x, 64, scope='conv1_1')
                features['Relu_1_1'] = x
                x = conv(x, 64, scope='conv1_2')
                x = slim.max_pool2d(x, [2, 2], scope='pool1')

                x = conv(x, 128, scope='conv2_1')
                features['Relu_2_1'] = x
                x = conv(x, 128, scope='conv2_2')
                x = slim.max_pool2d(x, [2, 2], scope='pool2')

                x = conv(x, 256, scope='conv3_1')
                features['Relu_3_1'] = x
                x = conv(x, 256, scope='conv3_2')
                x = conv(x, 256, scope='conv3_3')
                x = conv(x, 256, scope='conv3_4')
                x = slim.max_pool2d(x, [2, 2], scope='pool3')

                x = conv(x, 512, scope='conv4_1')
                features['Relu_4_1'] = x
                x = conv(x, 512, scope='conv4_2')
                x = conv(x, 512, scope='conv4_3')
                x = conv(x, 512, scope='conv4_4')
                x = slim.max_pool2d(x, [2, 2], scope='pool4')

                x = conv(x, 512, scope='conv5_1')
                features['Relu_5_1'] = x

    return features


def decoder(x, feature):
    """
    Arguments:
        x: a float tensor with shape [batch_size, height, width, depth],
            features from an encoder.
        feature: a string, name of a feature to decode.
            Possible values are: 'Relu_X_1', where X = 1, 2, 3, 4, 5.
    Returns:
        a float tensor with shape [batch_size, image_height, image_width, 3],
            it represents RGB images with pixel values in range [0, 255].
    """
    X = int(feature[5])

    with tf.variable_scope('decoder_' + str(X)):
        with slim.arg_scope([slim.conv2d], stride=1, padding='VALID'):

            if X > 4:
                x = conv(x, 512, scope='conv5_1')
                x = upsampling(x, scope='upsampling4')
                x = conv(x, 512, scope='conv4_4')
                x = conv(x, 512, scope='conv4_4')
                x = conv(x, 512, scope='conv4_2')
            if X > 3:
                x = conv(x, 256, scope='conv4_1')
                x = upsampling(x, scope='upsampling3')
                x = conv(x, 256, scope='conv3_4')
                x = conv(x, 256, scope='conv3_3')
                x = conv(x, 256, scope='conv3_2')
            if X > 2:
                x = conv(x, 128, scope='conv3_1')
                x = upsampling(x, scope='upsampling2')
                x = conv(x, 128, scope='conv2_2')
            if X > 1:
                x = conv(x, 64, scope='conv2_1')
                x = upsampling(x, scope='upsampling1')
                x = conv(x, 64, scope='conv1_2')
            if X > 0:
                x = conv(x, 3, use_relu=False, scope='conv1_1')
                # output is approximately in the range [-1, 1]

    with tf.name_scope('standardize'):
        image = 255.0 * 0.5 * (x + 1.0)
        # now output is in the range [0, 255]

    return image


def conv(x, channels, use_relu=True, scope='conv'):
    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    x = slim.conv2d(x, channels, [3, 3], activation_fn=tf.nn.relu if use_relu else None, scope=scope)
    return x


def upsampling(x, rate=2, scope='upsampling'):
    """Just a nearest neighbor upsampling."""
    with tf.name_scope(scope):

        shape = tf.shape(x)
        batch_size = shape[0]
        height, width = shape[1], shape[2]
        depth = x.shape[3].value

        x = tf.reshape(x, [batch_size, height, 1, width, 1, depth])
        x = tf.tile(x, [1, 1, rate, 1, rate, 1])
        x = tf.reshape(x, [batch_size, height * rate, width * rate, depth])

        return x
