import tensorflow as tf
from networks import encoder, decoder
tf.logging.set_verbosity('INFO')


"""
It creates a .pb frozen inference graph.
"""


GPU_TO_USE = '0'
FEATURE_TO_USE = 'Relu_1_1'
PB_FILE_PATH = 'model.pb'
CHECKPOINT = 'models/run00/model.ckpt-200000'


def convert_to_pb():

    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = GPU_TO_USE

    with graph.as_default():

        raw_images = tf.placeholder(dtype=tf.uint8, shape=[None, None, None, 3], name='images')
        encoding = tf.identity(encoder(tf.to_float(raw_images))[FEATURE_TO_USE], 'features')
        restored_images = tf.identity(decoder(encoding, FEATURE_TO_USE), 'restored_images')

        saver = tf.train.Saver()
        with tf.Session(graph=graph, config=config) as sess:
            saver.restore(sess, CHECKPOINT)

            # output ops
            keep_nodes = ['features', 'restored_images']

            input_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(),
                output_node_names=keep_nodes
            )
            output_graph_def = tf.graph_util.remove_training_nodes(
                input_graph_def, protected_nodes=keep_nodes
            )

            with tf.gfile.GFile(PB_FILE_PATH, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print('%d ops in the final graph.' % len(output_graph_def.node))


convert_to_pb()
