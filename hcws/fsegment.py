# encoding: utf-8
"""Reload and serve a saved model"""

import io
import json
import pickle

import numpy as np
import tensorflow as tf
from absl import flags, app
from pathlib import Path

from bert import modeling, tokenization
from input_fns import CwsProcessor
from model_fns import model_fn_builder

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', None, '')
flags.DEFINE_string('label_dict_file', None, '')
flags.DEFINE_string('data_dir', None, '')
flags.DEFINE_string('ckpt_dir', None, '')
flags.DEFINE_integer('batch_size', 100, '')


def create_feature_from_tokens(tokenizer, text_list, max_seq_length):
    # 保存label->index 的map
    tokens = []
    for word in text_list:
        # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），可以将所有的分字操作替换为list(input)
        token = tokenizer.tokenize(word)
        tokens.extend(token)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    ntokens = []
    segment_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)
    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        ntokens.append("**NULL**")

    return input_ids, input_mask, segment_ids


def text_input_fn_builder(input_file, tokenizer, max_seq_length):
    def _parse_fn(tokens):
        input_ids, input_mask, segment_ids = create_feature_from_tokens(tokenizer, tokens, max_seq_length)
        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids
        }

    def _generator_fn():
        for line in io.open(input_file, encoding='utf8'):
            yield _parse_fn(line.strip('\n'))

    def input_fn(params):
        batch_size = params["batch_size"]
        output_types = {"input_ids": tf.int64, "input_mask": tf.int64, "segment_ids": tf.int64}
        output_shapes = {
            "input_ids": [max_seq_length],
            "input_mask": [max_seq_length],
            "segment_ids": [max_seq_length]
        }
        d = tf.data.Dataset.from_generator(
            _generator_fn, output_types=output_types, output_shapes=output_shapes)
        d = d.batch(batch_size)

        return d

    return input_fn


def decode(tokens, labels):
    output = []
    prev_tag = None
    for t, l in zip(tokens, labels):
        if l.endswith('_B'):
            if prev_tag:
                output[-1] = u'{}/{}'.format(output[-1], prev_tag[:-2])
                output.append(' ')
        output.append(t)
        prev_tag = l
    if output:
        output[-1] = u'{}/{}'.format(output[-1], prev_tag[:-2])
    return output


def main(_):
    with Path(FLAGS.ckpt_dir, 'params.json').open(mode='r') as f:
        params = json.loads(f.read())
    with Path(FLAGS.data_dir, 'train.tf_record.json').open(mode='r') as f:
        train_data_info = json.loads(f.read())

    params['batch_size'] = FLAGS.batch_size

    processor = CwsProcessor(None)
    label_to_id = pickle.load(open(FLAGS.label_dict_file, 'rb'))
    id_to_label = dict([(v, k) for k, v in label_to_id.items()])
    label_list = processor.get_labels()

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.ckpt_dir,
        save_checkpoints_steps=1000,
        log_step_count_steps=100)
    tokenizer = tokenization.FullTokenizer(vocab_file=params['vocab_file'], do_lower_case=params['do_lower_case'])
    predict_input_fn = text_input_fn_builder(
        FLAGS.input_file,
        tokenizer,
        params['max_sequence_length'])
    # features = predict_input_fn(params).make_one_shot_iterator().get_next()
    # with tf.Session() as session:
    #     while True:
    #         try:
    #             batch_inputs = session.run(features)
    #             print(batch_inputs['input_ids'])
    #         except tf.errors.OutOfRangeError:
    #             break
    # return
    bert_config = modeling.BertConfig.from_json_file(params['bert_config_file'])
    num_warmup_steps = int(train_data_info['num_train_steps'] * params['warmup_proportion'])
    model_fn = model_fn_builder(
        bert_config,
        label_list,
        params['max_sequence_length'],
        train_data_info['num_train_steps'],
        num_warmup_steps,
        params['init_checkpoint'],
        use_ft=True)

    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)

    result = estimator.predict(input_fn=predict_input_fn)
    for (i, prediction) in enumerate(result):
        pred_ids = prediction['predictions']
        input_ids = prediction['input_ids']
        end = np.count_nonzero(input_ids)
        _ids = pred_ids[:end]
        _t_ids = input_ids[:end]
        labels = [id_to_label.get(_id, 'O') for _id in _ids[1:-1]]
        tokens = tokenizer.convert_ids_to_tokens(_t_ids[1:-1])
        output = decode(tokens, labels)
        print((u''.join(output)).encode('utf8'))


if __name__ == '__main__':
    flags.mark_flag_as_required('input_file')
    flags.mark_flag_as_required('label_dict_file')
    flags.mark_flag_as_required('data_dir')
    flags.mark_flag_as_required('ckpt_dir')
    try:
        app.run(main)
    except KeyboardInterrupt:
        print('User breaked .')
