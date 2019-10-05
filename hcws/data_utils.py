# -*- coding:utf-8 -*-
import json
import os
import tensorflow as tf
from absl import flags, app, logging

from input_fns import filed_based_convert_examples_to_features, CwsProcessor, file_based_input_fn_builder
from bert import tokenization

Q2B_DICT_FILE = '/home/heqing/git_repo/kcws/scripts/q2b.dic'

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", None, "")
flags.DEFINE_string("vocab_file", None, "")
flags.DEFINE_string("labels_file", None, "")
flags.DEFINE_string("output_dir", None, "")
flags.DEFINE_integer("max_seq_length", 200, "")
flags.DEFINE_integer("batch_size", 16, "")
flags.DEFINE_integer("num_train_epochs", 10, "")
flags.DEFINE_bool("do_lower_case", True, "")


def dump_dataset_info(output_dir, split, split_info):
    info_file = os.path.join(output_dir, split + ".tf_record.json")
    with open(info_file, 'w+') as info_fp:
        info_fp.write(json.dumps(split_info, ensure_ascii=False))


def main(_):
    processor = CwsProcessor(FLAGS.output_dir)
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    train_examples = processor.get_train_examples(FLAGS.data_dir)
    label_list = processor.get_labels()
    num_train_steps = int(
        len(train_examples) * 1.0 / FLAGS.batch_size * FLAGS.num_train_epochs)
    if num_train_steps < 1:
        raise AttributeError('training data is so small...')

    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_examples))
    logging.info("  Batch size = %d", FLAGS.batch_size)
    logging.info("  Num steps = %d", num_train_steps)

    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    if not os.path.exists(train_file):
        filed_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, FLAGS.output_dir)

    dump_dataset_info(
        FLAGS.output_dir,
        'train',
        {'num_examples': len(train_examples),
         'batch_size': FLAGS.batch_size,
         'num_train_steps': num_train_steps})

    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    # 打印验证集数据信息
    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", len(eval_examples))
    logging.info("  Batch size = %d", FLAGS.batch_size)

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    if not os.path.exists(eval_file):
        filed_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file, FLAGS.output_dir)

    dump_dataset_info(FLAGS.output_dir, 'eval',
                      {'num_examples': len(eval_examples),
                       'batch_size': FLAGS.batch_size})

    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    logging.info("***** Running prediction*****")
    logging.info("  Num examples = %d", len(predict_examples))
    logging.info("  Batch size = %d", FLAGS.batch_size)

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    if not os.path.exists(predict_file):
        filed_based_convert_examples_to_features(predict_examples, label_list,
                                                 FLAGS.max_seq_length, tokenizer,
                                                 predict_file, FLAGS.output_dir, mode="test")

    dump_dataset_info(FLAGS.output_dir, 'test',
                      {'num_examples': len(predict_examples),
                       'batch_size': FLAGS.batch_size})


class TestDataUtils(tf.test.TestCase):
    def testFileInputFn(self):
        max_seq_len = 122
        train_file = os.path.join('./output', 'train.tf_record')
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=max_seq_len,
            is_training=True,
            drop_remainder=True)
        params = {
            'batch_size': 16
        }
        features = train_input_fn(params).make_one_shot_iterator().get_next()
        input_ids = features['input_ids']
        label_ids = features['label_ids']
        used = tf.sign(tf.abs(input_ids))
        lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度
        weights = tf.sequence_mask(lengths, max_seq_len)
        # is_correct = tf.to_float(tf.equal(predictions, labels))
        with self.test_session() as session:
            while True:
                try:
                    inputs_np, labels_np, weights_np = session.run([input_ids, label_ids, weights])
                    print("inputs.shape: {}\tlabels.shape: {}\tweights.shape: {}".format(
                        inputs_np.shape, labels_np.shape, weights_np.shape))
                except tf.errors.OutOfRangeError:
                    break

        print('test file input fn')


if __name__ == '__main__':
    flags.mark_flag_as_required('data_dir')
    flags.mark_flag_as_required('vocab_file')
    flags.mark_flag_as_required('output_dir')
    app.run(main)
    # tf.test.main()
