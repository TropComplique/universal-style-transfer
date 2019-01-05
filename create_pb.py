import tensorflow as tf
from networks import encoder, decoder
tf.logging.set_verbosity('INFO')


"""
It creates .pb frozen inference graphs.
"""


GPU_TO_USE = '0'
NUM_FEATURES = {1: 64, 2: 128, 3: 256, 4: 512, 5: 512}
CHECKPOINT = {
    1: 'models/run00/model.ckpt-200000',
    2: 'models/run01/model.ckpt-200000',
    3: 'models/run02/model.ckpt-200000',
    4: 'models/run03/model.ckpt-200000',
    5: 'models/run04/model.ckpt-200000'
}


def convert_decoder_to_pb(X):

    features_to_use = 'Relu_{}_1'.format(X)
    pb_file_path = 'inference/decoder_{}.pb'.format(X)

    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = GPU_TO_USE

    with graph.as_default():

        features = tf.placeholder(
            dtype=tf.float32, name='features',
            shape=[None, None, None, NUM_FEATURES[X]]
        )
        restored_images = tf.identity(
            decoder(features, features_to_use),
            'restored_images'
        )

        saver = tf.train.Saver()
        with tf.Session(graph=graph, config=config) as sess:
            saver.restore(sess, CHECKPOINT[X])

            # output ops
            keep_nodes = ['restored_images']

            input_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(),
                output_node_names=keep_nodes
            )
            output_graph_def = tf.graph_util.remove_training_nodes(
                input_graph_def, protected_nodes=keep_nodes
            )

            with tf.gfile.GFile(pb_file_path, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print('%d ops in the final graph.' % len(output_graph_def.node))


def convert_encoder_to_pb():

    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = GPU_TO_USE

    with graph.as_default():

        raw_images = tf.placeholder(dtype=tf.uint8, shape=[None, None, None, 3], name='images')
        names = ['Relu_{}_1'.format(X) for X in range(1, 6)]
        features = encoder(tf.to_float(raw_images))
        features = [tf.identity(features[n], n) for n in names]
        tf.train.init_from_checkpoint(
            'pretrained/vgg_19.ckpt',
            {'vgg_19/': 'encoder/'}
        )

        with tf.Session(graph=graph, config=config) as sess:
            sess.run(tf.global_variables_initializer())

            # output ops
            keep_nodes = names

            input_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(),
                output_node_names=keep_nodes
            )
            output_graph_def = tf.graph_util.remove_training_nodes(
                input_graph_def, protected_nodes=keep_nodes
            )

            with tf.gfile.GFile('inference/encoder.pb', 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print('%d ops in the final graph.' % len(output_graph_def.node))


convert_encoder_to_pb()
for X in [1, 2, 3, 4, 5]:
    convert_decoder_to_pb(X)
