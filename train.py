import tensorflow as tf
import os
from model import model_fn
from pipeline import Pipeline

tf.logging.set_verbosity('INFO')


"""
The purpose of this script is to train a detector.
Evaluation will happen periodically.

To use it just run:
python train.py
"""

GPU_TO_USE = '0'

params = {
    'lambda': 1.0,
    'feature_to_use': 'Relu_5_1',  # 'Relu_X_1'
    'train_dataset': '/mnt/datasets/COCO/ust/train/',
    'val_dataset': '/mnt/datasets/COCO/ust/val/',
    'batch_size': 8,
    'model_dir': 'models/run00/',
    'num_steps': 16000,
    'pretrained_checkpoint': 'pretrained/vgg_19.ckpt',
    'weight_decay': 1e-8,
}


def get_input_fn(is_training):

    dataset_path = params['train_dataset'] if is_training else params['val_dataset']
    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]
    batch_size = params['batch_size'] if is_training else 1

    def input_fn():
        pipeline = Pipeline(filenames, is_training, batch_size)
        return pipeline.dataset

    return input_fn


session_config = tf.ConfigProto()
session_config.gpu_options.visible_device_list = GPU_TO_USE
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=params['model_dir'], session_config=session_config,
    save_summary_steps=200, save_checkpoints_secs=600,
    log_step_count_steps=100
)

train_input_fn = get_input_fn(is_training=True)
val_input_fn = get_input_fn(is_training=False)
estimator = tf.estimator.Estimator(model_fn, params=params, config=run_config)

train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=params['num_steps'])
eval_spec = tf.estimator.EvalSpec(val_input_fn, steps=None, start_delay_secs=3600, throttle_secs=3600)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
