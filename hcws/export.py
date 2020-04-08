# encoding: utf-8
"""Export model as a saved_model"""

import json

import tensorflow as tf
from pathlib import Path

from bert import modeling
from input_fns import get_task_processor
from model_fns import model_fn_builder

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("task_name", None, "")
tf.flags.DEFINE_string("train_data_path", None, "")
tf.flags.DEFINE_string("ckpt_dir", None, "")
tf.flags.DEFINE_string("export_dir", None, "")


def main(_):
    processor = get_task_processor(FLAGS.task_name, None)
    label_list = processor.get_labels()
    with Path(FLAGS.ckpt_dir, 'params.json').open(mode='r') as f:
        params = json.loads(f.read())

    max_sequence_length = params['max_sequence_length']

    bert_config = modeling.BertConfig.from_json_file(params['bert_config_file'])
    with open("{}.json".format(FLAGS.train_data_path), 'r') as f:
        train_info = json.load(f)

    model_fn = model_fn_builder(
        bert_config,
        label_list,
        max_sequence_length,
        num_train_steps=train_info['num_train_steps'],
        num_warmup_steps=int(train_info['num_train_steps'] * params['warmup_proportion']),
        init_checkpoint=params['init_checkpoint'],
        use_crf=params['use_crf'])

    estimator = tf.estimator.Estimator(model_fn, FLAGS.ckpt_dir, params=params)
    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        # 'label_ids': label_ids,
        'input_ids': tf.placeholder(tf.int32, [None, max_sequence_length], name='input_ids'),
        'input_mask': tf.placeholder(tf.int32, [None, max_sequence_length], name='input_mask'),
        'segment_ids': tf.placeholder(tf.int32, [None, max_sequence_length], name='segment_ids'),
    })
    estimator.export_saved_model(FLAGS.export_dir, serving_input_receiver_fn)


if __name__ == '__main__':
    tf.app.run()
