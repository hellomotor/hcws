# encoding: utf-8
"""Reload and serve a saved model"""
import io
import numpy as np
from pathlib import Path
from tensorflow.contrib import predictor
import sys
from bert import tokenization
from absl import flags, app
import json
import os
import pickle
from utils import create_feature_from_tokens, label_decode

HOME = os.path.expanduser('~')

Q2B_DICT_FILE = os.path.join(HOME, 'git_repo/hcws/scripts/q2b.dic')

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', None, '')
flags.DEFINE_string('ckpt_dir', None, '')
flags.DEFINE_string('export_dir', None, '')
flags.DEFINE_string('q2b_file', Q2B_DICT_FILE, '')


def main(_):
    with Path(FLAGS.ckpt_dir, 'params.json').open(mode='r') as f:
        params = json.loads(f.read())

    label_to_id = pickle.load(open(os.path.join(FLAGS.data_dir, 'label2id.pkl'), 'rb'))
    id_to_label = dict([(v, k) for k, v in label_to_id.items()])
    q2b_dict = dict([line.strip('\n').split('\t') for line in io.open(FLAGS.q2b_file, encoding='utf8')])

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
        feature = create_feature_from_tokens(tokenizer, q2b_dict, [words], 200)
        predictions = predict_fn({
            'input_ids': feature['input_ids'],
            'input_mask': feature['input_mask'],
            'segment_ids': feature['segment_ids']
            })
        end = len(words) + 2 # [CLS] ... [SEP]
        pred_ids = predictions['predictions'][0][:end]
        labels = [id_to_label.get(_id, 'O') for _id in pred_ids[1:-1]]
        output = label_decode(words, labels)
        print((u''.join(output)).encode('utf8'))


if __name__ == '__main__':
    flags.mark_flag_as_required('data_dir')
    flags.mark_flag_as_required('ckpt_dir')
    flags.mark_flag_as_required('export_dir')
    app.run(main)

