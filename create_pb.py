import tensorflow as tf
from networks import encoder, decoder
tf.logging.set_verbosity('INFO')


"""
It creates a .pb frozen inference graph.
"""


GPU_TO_USE = '0'
FEATURE_TO_USE = 'Relu_5_1'
NUM_FEATURES = 512 # 64, 128, 256, 512, 512
PB_FILE_PATH = 'decoder_5.pb'
CHECKPOINT = 'models/run04/model.ckpt-200000'


def convert_to_pb():

    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = GPU_TO_USE

    with graph.as_default():

        features = tf.placeholder(dtype=tf.float32, shape=[None, None, None, NUM_FEATURES], name='features')
        restored_images = tf.identity(decoder(features, FEATURE_TO_USE), 'restored_images')

        saver = tf.train.Saver()
        with tf.Session(graph=graph, config=config) as sess:
            saver.restore(sess, CHECKPOINT)

            # output ops
            keep_nodes = ['restored_images']

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

            with tf.gfile.GFile('encoder.pb', 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print('%d ops in the final graph.' % len(output_graph_def.node))


convert_to_pb()
convert_encoder_to_pb()
