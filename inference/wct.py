def wct(content, style, alpha, epsilon=1e-8):
    """
    Arguments:
        content: a float tensor with shape [1, H_c, W_c, C].
        style: a float tensor with shape [1, H_s, W_s, C].
        alpha, eps: float numbers.
    Returns:
        a float tensor with shape [1, H_c, W_c, C].
    """
    _, H_c, W_c, C = tf.unstack(tf.shape(content), axis=0)
    _, H_s, W_s, _ = tf.unstack(tf.shape(style), axis=0)
    
    content = tf.transpose(tf.squeeze(content, 0), [2, 0, 1])
    content = tf.reshape(content, [C, H_c * W_c])
    style = tf.transpose(tf.squeeze(style, 0), [2, 0, 1])
    style = tf.reshape(style, [C, H_s * W_s])

    content_mean = tf.reduce_mean(content, axis=1, keepdims=True)
    fc = content - content_mean
    content_covariance = tf.matmul(fc, fc, transpose_b=True) / (tf.to_float(H_c * W_c) - 1.0)
    content_covariance += tf.eye(C) * epsilon

    style_mean = tf.reduce_mean(style, axis=1, keepdims=True)
    fs = style - style_mean
    style_covariance = tf.matmul(fs, fs, transpose_b=True) / (tf.to_float(H_s * W_s) - 1.0)
    style_covariance += tf.eye(C) * epsilon

    with tf.device('/cpu:0'):  
        D_c, E_c, _ = tf.svd(content_covariance)
        D_s, E_s, _ = tf.svd(style_covariance)

    # filter small singular values
    k_c = tf.reduce_sum(tf.to_int32(tf.greater(D_c, 1e-5)), 0)
    k_s = tf.reduce_sum(tf.to_int32(tf.greater(D_s, 1e-5)), 0)
    D_c, E_c = D_c[:k_c], E_c[:, :k_c]
    D_s, E_s = D_s[:k_s], E_s[:, :k_s]

    # whiten content features
    x = tf.matmul(E_c, tf.diag(tf.pow(D_c, -0.5)))
    fc_hat = tf.matmul(tf.matmul(x, E_c, transpose_b=True), fc)

    # color content with style
    x = tf.matmul(E_s, tf.diag(tf.pow(D_s, 0.5)))
    fcs_hat = tf.matmul(tf.matmul(x, E_s, transpose_b=True), fc_hat)

    fcs_hat = fcs_hat + style_mean

    # blend whiten-colored feature with original content feature
    blended = alpha * fcs_hat + (1.0 - alpha) * content

    blended = tf.reshape(blended, [C, H_c, W_c])
    blended = tf.expand_dims(tf.transpose(blended, (1, 2, 0)), 0)
    return blended


class Transfer:
    def __init__(self, X):

        with tf.gfile.GFile('encoder.pb', 'rb') as f:
            graph_def1 = tf.GraphDef()
            graph_def1.ParseFromString(f.read())

        with tf.gfile.GFile('decoder_{}.pb'.format(X), 'rb') as f:
            graph_def2 = tf.GraphDef()
            graph_def2.ParseFromString(f.read())

        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def1, name='encoder')
            tf.import_graph_def(graph_def2, name='decoder')

            image = graph.get_tensor_by_name('encoder/images:0')
            features = graph.get_tensor_by_name('decoder/features:0')
            encoding = graph.get_tensor_by_name('encoder/Relu_{}_1:0'.format(X))
            style = tf.placeholder(tf.float32, [None, None, None, encoding.shape[3].value])
            alpha = tf.placeholder(tf.float32, [])
            
            self.input_tensors = {
                'features': features,
                'image': image,
                'style': style, 'alpha': alpha
            }
            
            blended = wct(encoding, style, alpha=alpha, epsilon=1e-8)
            self.output_tensors = {
                'encoding': encoding,
                'restored_images': graph.get_tensor_by_name('decoder/restored_images:0'),
                'blended': blended
            }

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=1.0,
            visible_device_list='0'
        )
        config_proto = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        self.sess = tf.Session(graph=graph, config=config_proto)
        
    def get_features(self, image):
        return self.sess.run(
            self.output_tensors['encoding'], 
            {self.input_tensors['image']: np.expand_dims(image, 0)}
        )
    
    def decode(self, features):
        return self.sess.run(
            self.output_tensors['restored_images'], 
            {self.input_tensors['features']: features}
        )[0]
    
    def blend(self, image, style, alpha):
        return self.sess.run(
            self.output_tensors['blended'], 
            {self.input_tensors['image']: np.expand_dims(image, 0), 
             self.input_tensors['style']: style,
             self.input_tensors['alpha']: alpha}
        )