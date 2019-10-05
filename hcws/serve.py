# encoding: utf-8
"""Reload and serve a saved model"""

import numpy as np
from pathlib import Path
from tensorflow.contrib import predictor
import sys
from bert import tokenization
from absl import flags, app
import json
import os
import pickle

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', None, '')
flags.DEFINE_string('ckpt_dir', None, '')
flags.DEFINE_string('export_dir', None, '')


def create_feature_from_tokens(tokenizer, text_list, params):
    max_seq_length = params['max_sequence_length']
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
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return {
        'input_ids': np.array([input_ids], dtype=np.int64),
        'input_mask': np.array([input_mask], dtype=np.int64),
        'segment_ids': np.array([segment_ids], dtype=np.int64),
    }


def main(_):
    with Path(FLAGS.ckpt_dir, 'params.json').open(mode='r') as f:
        params = json.loads(f.read())

    label_to_id = pickle.load(open(os.path.join(FLAGS.data_dir, 'label2id.pkl'), 'rb'))
    id_to_label = dict([(v, k) for k, v in label_to_id.items()])

    tokenizer = tokenization.FullTokenizer(vocab_file=params['vocab_file'], do_lower_case=params['do_lower_case'])
    subdirs = [x for x in Path(FLAGS.export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(latest)
    while True:
        sys.stdout.write('=> ')
        sys.stdout.flush()
        inputs = sys.stdin.readline().strip('\n').decode('utf8')
        if inputs.lower() == 'quit':
            break
        words = [w for w in inputs]
        feature = create_feature_from_tokens(tokenizer, words, params)
        predictions = predict_fn({
            'input_ids': feature['input_ids'],
            'input_mask': feature['input_mask'],
            'segment_ids': feature['segment_ids']
            })
        end = len(words) + 2 # [CLS] ... [SEP]
        pred_ids = predictions['output'][0][:end]
        labels = [id_to_label.get(_id, 'O') for _id in pred_ids[1:-1]]
        output = []
        prev_tag = None
        for t, l in zip(words, labels):
            if l.endswith('_B'):
                if prev_tag:
                    output[-1] = u'{}/{}'.format(output[-1], prev_tag[:-2])
                    output.append(' ')
            output.append(t)
            prev_tag = l
        if output:
            output[-1] = u'{}/{}'.format(output[-1], prev_tag[:-2])
        print((u''.join(output)).encode('utf8'))


if __name__ == '__main__':
    flags.mark_flag_as_required('data_dir')
    flags.mark_flag_as_required('ckpt_dir')
    flags.mark_flag_as_required('export_dir')
    app.run(main)

