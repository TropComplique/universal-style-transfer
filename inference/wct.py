import tensorflow as tf


EPSILON = 1e-8


class Transfer:

    def get_features(self, image):
        """
        Extracts all 'Relu_X_1' features from an image.

        Arguments:
            image: a uint8 numpy array with shape [h, w, 3].
        Returns:
            a dict with float numpy arrays.
        """
        feed_dict = {self.input_tensors['image']: np.expand_dims(image, 0)}
        return self.sess.run(self.output_tensors['encodings'], feed_dict)

    def decode(self, features, X):
        """
        Predicts an image from 'Relu_X_1' features.

        Arguments:
            features: a float numpy array with shape [1, H, W, C].
            X: an integer.
        Returns:
            a float numpy array with shape [h, w, 3].
        """
        feed_dict = {self.input_tensors['features'][X]: features}
        return self.sess.run(self.output_tensors['restored_images'][X], feed_dict)[0]

    def get_style(self, features):
        """
        Extracts style transforms from 'Relu_X_1' features.

        Arguments:
            features: a dict with float numpy arrays.
        Returns:
            a dict with float numpy arrays.
        """
        feed_dict = {
            self.input_tensors['features'][X]: features[X]
            for X in features
        }
        output = {
            X: self.output_tensors['style_transforms'][X]
            for X in features
        }
        return self.sess.run(output, feed_dict)

    def blend(self, features, X, style_mean, coloring_matrix, alpha):
        """
        Arguments:
            features: a float numpy array with shape [1, H, W, C].
            X: an integer.
            style_mean: a float numpy array with shape [C].
            coloring_matrix: a float numpy array with shape [C, C].
            alpha: a float number.
        Returns:
            a float numpy array with shape [1, H, W, C].
        """
        feed_dict = {
            self.input_tensors['features'][X]: input_features,
            self.input_tensors['style_mean']: style_mean,
            self.input_tensors['coloring_matrix']: coloring_matrix
        }
        return self.sess.run(self.output_tensors['blended'][X], feed_dict)

    def __init__(self):

        features_to_use = [1, 2, 3, 4, 5]
        graph = tf.Graph()
        with graph.as_default():

            # LOAD ALL GRAPHS

            with tf.gfile.GFile('encoder.pb', 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='encoder')

            for X in features_to_use:
                with tf.gfile.GFile('decoder_{}.pb'.format(X), 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name='decoder_{}'.format(X))

            # CREATE INPUT TENSORS

            input_image = graph.get_tensor_by_name('encoder/images:0')
            input_features = {
                X: graph.get_tensor_by_name('decoder_{}/features:0'.format(X))
                for X in features_to_use
            }
            style_mean = tf.placeholder(tf.float32, [None, 1])
            coloring_matrix = tf.placeholder(tf.float32, [None, None])
            self.input_tensors = {
                'image': input_image,
                'features': input_features,
                'style_mean': style_mean,
                'coloring_matrix': coloring_matrix
            }

            # CREATE STYLE TRANSFER GRAPH

            style_transforms = {
                X: get_style_transform(input_features[X])
                for X in features_to_use
            }
            blended_features = {
                X: wct(input_features[X], style_mean, coloring_matrix, alpha)
                for X in features_to_use
            }

            # CREATE OUTPUT TENSORS

            encodings = {
                X: graph.get_tensor_by_name('encoder/Relu_{}_1:0'.format(X))
                for X in features_to_use
            }
            decodings = {
                X: graph.get_tensor_by_name('decoder_{}/restored_images:0'.format(X))
                for X in features_to_use
            }
            self.output_tensors = {
                'encodings': encodings,
                'restored_images': decodings,
                'blended': blended_features,
                'style_transforms': style_transforms
            }

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=1.0,
            visible_device_list='0'
        )
        config_proto = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        self.sess = tf.Session(graph=graph, config=config_proto)


def extract_statistics(features):
    """
    Arguments:
        features: a float tensor with shape [1, H, W, C].
    Returns:
        f: a float tensor with shape [C, H * W].
        mean: a float tensor with shape [C, 1].
        covariance: a float tensor with shape [C, C].
    """
    features = tf.transpose(tf.squeeze(features, 0), [2, 0, 1])
    C, H, W = tf.unstack(tf.shape(features), axis=0)
    features = tf.reshape(features, [C, H * W])

    mean = tf.reduce_mean(features, axis=1, keepdims=True)  # shape [C, 1]
    f = features - mean  # centered features
    covariance = tf.matmul(f, f, transpose_b=True) / (tf.to_float(H * W) - 1.0)
    covariance += tf.eye(C) * EPSILON

    return f, mean, covariance


def get_style_transform(features):
    """
    Arguments:
        features: a float tensor with shape [1, H, W, C].
    Returns:
        style_mean: a float tensor with shape [C, 1].
        coloring_matrix: a float tensor with shape [C, C].
    """
    _, style_mean, style_covariance = extract_statistics(features)
    D_s, E_s, _ = tf.svd(style_covariance)

    # filter small singular values
    k_s = tf.reduce_sum(tf.to_int32(tf.greater(D_s, 1e-5)), 0)
    D_s, E_s = D_s[:k_s], E_s[:, :k_s]
    # they have shapes [k_s] and [C, k_s]

    x = tf.matmul(E_s, tf.diag(tf.pow(D_s, 0.5)))  # shape [C, k_s]
    coloring_matrix = tf.matmul(x, E_s, transpose_b=True)  # shape [C, C]
    return style_mean, coloring_matrix


def wct(content, style_mean, coloring_matrix, alpha):
    """
    Arguments:
        content: a float tensor with shape [1, H_c, W_c, C].
        style_mean: a float tensor with shape [C].
        coloring_matrix: a float tensor with shape [C, C].
        alpha: a float number.
    Returns:
        a float tensor with shape [1, H_c, W_c, C].
    """

    fc, _, content_covariance = extract_style(content)
    # they have shapes [C, H_c * W_c] and [C, C]

    with tf.device('/cpu:0'):
        D_c, E_c, _ = tf.svd(content_covariance)

    # filter small singular values
    k_c = tf.reduce_sum(tf.to_int32(tf.greater(D_c, 1e-5)), 0)
    D_c, E_c = D_c[:k_c], E_c[:, :k_c]
    # they have shapes [k_c] and [C, k_c]

    # whiten content features
    x = tf.matmul(E_c, tf.diag(tf.pow(D_c, -0.5)))  # shape [C, k_c]
    fc_hat = tf.matmul(tf.matmul(x, E_c, transpose_b=True), fc)  # shape [C, H_c * W_c]

    fcs_hat = tf.matmul(coloring_matrix, fc_hat)  # shape [C, H_c * W_c]
    fcs_hat += style_mean

    # blend whiten-colored feature with original content feature
    blended = alpha * fcs_hat + (1.0 - alpha) * content

    blended = tf.reshape(blended, [C, H_c, W_c])
    blended = tf.expand_dims(tf.transpose(blended, (1, 2, 0)), 0)
    return blended
