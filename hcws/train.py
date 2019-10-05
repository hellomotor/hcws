# encoding: utf-8

import json

import tensorflow as tf
import os

from bert import modeling
from input_fns import file_based_input_fn_builder, CwsProcessor
from model_fns import model_fn_builder

flags = tf.flags

flags.DEFINE_string("train_data_path", default=None, help="")
flags.DEFINE_string("test_data_path", default=None, help="")
flags.DEFINE_string("label_dict_path", default=None, help="")

flags.DEFINE_integer("batch_size", default=128, help="")
flags.DEFINE_float("learning_rate", default=1e-5, help="")
flags.DEFINE_float("dropout", default=0.5, help="")
flags.DEFINE_integer("max_sequence_length", default=200, help="")

flags.DEFINE_string("ckpt_dir", default=None, help="")
flags.DEFINE_integer("save_checkpoints_steps", default=1000, help="")
flags.DEFINE_integer("log_step_count_steps", default=100, help="")
flags.DEFINE_integer("early_stopping_step", default=4,
                     help="Stop training after #early_stopping_step when loss not decrease")

# bert config
tf.app.flags.DEFINE_string("bert_config_file", default=None, help="")
tf.app.flags.DEFINE_string("vocab_file", default=None, help="")
tf.app.flags.DEFINE_string("init_checkpoint", default=None, help="")
tf.app.flags.DEFINE_bool("do_lower_case", default=True, help="")
# train parameters
flags.DEFINE_float("warmup_proportion", default=0.1, help="")

FLAGS = flags.FLAGS


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    processor = CwsProcessor(None)
    params = FLAGS.flag_values_dict()
    label_list = processor.get_labels()
    with open("{}.json".format(FLAGS.train_data_path)) as f:
        train_data_info = json.load(f)

    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    config = tf.estimator.RunConfig(
        model_dir=FLAGS.ckpt_dir,
        session_config=session_config,
        keep_checkpoint_max=2,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        log_step_count_steps=FLAGS.log_step_count_steps)

    with open(os.path.join(FLAGS.ckpt_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    num_warmup_steps = int(train_data_info['num_train_steps'] * FLAGS.warmup_proportion)
    model_fn = model_fn_builder(
        bert_config,
        label_list,
        FLAGS.max_sequence_length,
        train_data_info['num_train_steps'],
        num_warmup_steps,
        FLAGS.init_checkpoint)

    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=config,
    )
    # early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
    #     estimator, "f1", 500, min_steps=8000, run_every_secs=120)
    # features = train_input_fn(params).make_one_shot_iterator().get_next()
    # input_ids, label_ids = features['input_ids'], features['label_ids']
    # with tf.Session(config=session_config) as session:
    #     batch_inputs, batch_labels = session.run([input_ids, label_ids])
    #     print(batch_inputs[0])
    #     print(batch_labels[0])
    # return

    train_input_fn = file_based_input_fn_builder(
        FLAGS.train_data_path,
        FLAGS.max_sequence_length,
        is_training=True,
        drop_remainder=True)

    eval_input_fn = file_based_input_fn_builder(
        FLAGS.test_data_path,
        FLAGS.max_sequence_length,
        is_training=False,
        drop_remainder=False)
    try:
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_data_info['num_train_steps'])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    except KeyboardInterrupt:
        tf.logging.info('User interrupted.')


if __name__ == '__main__':
    tf.app.run()
