import tensorflow as tf
from networks import encoder, decoder


def model_fn(features, labels, mode, params, config):
    """
    This is a function for creating a computational tensorflow graph.
    The function is in format required by tf.estimator.
    """
    images = features

    # build the main graph
    feature_to_use = params['feature_to_use']  # Relu_X_1
    encoding = encoder(images)[feature_to_use]
    restored_images = decoder(encoding, feature_to_use)
    encoding_of_restored_images = encoder(restored_images)[feature_to_use]

    is_training = mode == tf.estimator.ModeKeys.TRAIN

    # use a pretrained backbone network
    if is_training:
        with tf.name_scope('init_from_checkpoint'):
            tf.train.init_from_checkpoint(
                params['pretrained_checkpoint'],
                {'vgg_19/': 'encoder/'}
            )

    assert mode != tf.estimator.ModeKeys.PREDICT

    # add L2 regularization
    with tf.name_scope('weight_decay'):
        add_weight_decay(params['weight_decay'])
        regularization_loss = tf.losses.get_regularization_loss()

    batch_size = tf.to_float(tf.shape(images)[0])
    normalizer = 255.0 * batch_size
    reconstruction_loss = tf.nn.l2_loss(images - restored_images)/normalizer
    features_loss = tf.nn.l2_loss(encoding - encoding_of_restored_images)/normalizer

    tf.losses.add_loss(reconstruction_loss)
    tf.losses.add_loss(params['lambda'] * features_loss)
    tf.summary.scalar('regularization_loss', regularization_loss)
    tf.summary.scalar('reconstruction_loss', reconstruction_loss)
    tf.summary.scalar('features_loss', features_loss)

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    if mode == tf.estimator.ModeKeys.EVAL:

        eval_metric_ops = {
            'val_reconstruction_loss': tf.metrics.mean(reconstruction_loss),
            'val_features_loss': tf.metrics.mean(features_loss)
        }

        return tf.estimator.EstimatorSpec(
            mode, loss=total_loss,
            eval_metric_ops=eval_metric_ops
        )

    assert mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('learning_rate'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.polynomial_decay(
            params['initial_learning_rate'], global_step,
            params['num_steps'], params['end_learning_rate'],
            power=1.0  # linear decay
        )
        tf.summary.scalar('learning_rate', learning_rate)

    with tf.variable_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)
        train_op = optimizer.minimize(total_loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def add_weight_decay(weight_decay):
    weights = [v for v in tf.trainable_variables() if 'weights' in v.name]
    for w in weights:
        value = tf.multiply(weight_decay, tf.nn.l2_loss(w))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value)
