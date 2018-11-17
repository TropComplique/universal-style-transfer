import tensorflow as tf
from networks import encoder, decoder


def model_fn(features, labels, mode, params, config):
    """
    This is a function for creating a computational tensorflow graph.
    The function is in format required by tf.estimator.
    """
    images = features

    feature_to_use = params['feature_to_use']  # Relu_X_1
    encoding = encoder(images)[feature_to_use]

    restored_images = decoder(encoding, feature_to_use)
    encoding_of_restored_images = encoder(restored_images)[feature_to_use]

    # build the main graph
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    # use a pretrained backbone network
    if is_training:
        with tf.name_scope('init_from_checkpoint'):
            tf.train.init_from_checkpoint(
                params['pretrained_checkpoint'],
                {checkpoint_scope: checkpoint_scope}
            )

    # add NMS to the graph
    if not is_training:
        predictions = detector.get_predictions(
            score_threshold=params['score_threshold'],
            iou_threshold=params['iou_threshold'],
            max_boxes_per_class=params['max_boxes_per_class']
        )

    if mode == tf.estimator.ModeKeys.PREDICT:
        # this is required for exporting a savedmodel
        export_outputs = tf.estimator.export.PredictOutput({
            name: tf.identity(tensor, name)
            for name, tensor in predictions.items()
        })
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions,
            export_outputs={'outputs': export_outputs}
        )

    # add L2 regularization
    with tf.name_scope('weight_decay'):
        add_weight_decay(params['weight_decay'])
        regularization_loss = tf.losses.get_regularization_loss()

    reconstruction_loss = tf.nn.l2_loss(images - restored_images)
    features_loss = tf.nn.l2_loss(encoding - encoding_of_restored_images)

    tf.losses.add_loss(reconstruction_loss)
    tf.losses.add_loss(params['lambda'] * features_loss)
    tf.summary.scalar('reconstruction_loss', reconstruction_loss)
    tf.summary.scalar('features_loss', features_loss)

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    if mode == tf.estimator.ModeKeys.EVAL:

        with tf.name_scope('evaluator'):
            evaluator = Evaluator(class_names=['face'])
            eval_metric_ops = evaluator.get_metric_ops(labels, predictions)

        return tf.estimator.EstimatorSpec(
            mode, loss=total_loss,
            eval_metric_ops=eval_metric_ops
        )

    assert mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('learning_rate'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.piecewise_constant(global_step, params['lr_boundaries'], params['lr_values'])
        tf.summary.scalar('learning_rate', learning_rate)

    with tf.variable_scope('optimizer'):

        learning_rate = torch_decay(learning_rate, global_step, lr_decay)
        d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)

        # Only train decoder vars, encoder is frozen
        d_vars = [var for var in tf.trainable_variables() if 'decoder_'+relu_target in var.name]

        train_op = d_optimizer.minimize(total_loss, var_list=d_vars, global_step=global_step)
        batch_size=8,
        feature_weight=1,
        pixel_weight=1,
        tv_weight=0,
        learning_rate=1e-4,
        lr_decay=5e-5, # 0
        max_steps=16000
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        grads_and_vars = [(tf.clip_by_norm(g, 10.0), v) for g, v in grads_and_vars]
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)
        # train_op = tf.group(train_op)  # WTF!

    for g, v in grads_and_vars:
        tf.summary.histogram(v.name[:-2] + '_hist', v)
        tf.summary.histogram(v.name[:-2] + '_grad_hist', g)

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def add_weight_decay(weight_decay):
    weights = [
        v for v in tf.trainable_variables()
        if 'weights' in v.name and 'depthwise_weights' not in v.name
    ]
    for w in weights:
        value = tf.multiply(weight_decay, tf.nn.l2_loss(w))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value)
