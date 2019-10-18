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
from hcws.utils import create_feature_from_tokens, label_decode, batch_read_iter

HOME = os.path.expanduser('~')
BERT_BASE = os.path.join(HOME, 'git_repo/tf_ner/bert_lstm_crf/chinese_L-12_H-768_A-12')
CWD = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CWD)
LABEL_DICT_FILE = os.path.join(ROOT_DIR, 'corpus/RMRB/train/label2id.pkl')
SAVED_MODEL_DIR = os.path.join(ROOT_DIR, 'models/saved_model/hcws')
Q2B_DICT_FILE = os.path.join(HOME, 'git_repo/hcws/scripts/q2b.dic')

FLAGS = flags.FLAGS
flags.DEFINE_string('bert_vocab_file', os.path.join(BERT_BASE, 'vocab.txt'), '')
flags.DEFINE_string('label_file', LABEL_DICT_FILE, '')
flags.DEFINE_string('saved_model_dir', SAVED_MODEL_DIR, '')
flags.DEFINE_string('q2b_file', Q2B_DICT_FILE, '')
flags.DEFINE_string('input_file', None, '')
flags.DEFINE_integer('batch_size', 16, '')
flags.DEFINE_integer('max_sequence_length', 200, '')


def main(_):
    label_to_id = pickle.load(open(FLAGS.label_file, 'rb'))
    id_to_label = dict([(v, k) for k, v in label_to_id.items()])
    q2b_dict = dict([line.strip('\n').split('\t') for line in io.open(FLAGS.q2b_file, encoding='utf8')])

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.bert_vocab_file, do_lower_case=True)
    subdirs = [x for x in Path(FLAGS.saved_model_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(latest)
    for batch_of_line in batch_read_iter(FLAGS.input_file, FLAGS.batch_size):
        batch_of_tokens = [(w for w in line.strip('\n')[:FLAGS.max_sequence_length])
                           for line in batch_of_line]
        feature = create_feature_from_tokens(
            tokenizer, 
            q2b_dict, 
            batch_of_tokens, 
            FLAGS.max_sequence_length,
            ignore_space=False)
        predictions = predict_fn({
            'input_ids': feature['input_ids'],
            'input_mask': feature['input_mask'],
            'segment_ids': feature['segment_ids']
            })

        seq_lens = predictions['seq_lens']
        pred_ids = predictions['predictions']
        batch_size = min(len(batch_of_line), pred_ids.shape[0])
        input_ids = feature['input_ids']
        for i in range(batch_size):
            end = seq_lens[i]  # [CLS] ... [SEP]
            label_ids = pred_ids[i][:end]
            token_ids = input_ids[i][:end]
            labels = [id_to_label.get(_id, 'O') for _id in label_ids[1:-1]]
            tokens = tokenizer.convert_ids_to_tokens(token_ids[1:-1])
            output = label_decode(tokens, labels, output_pos=True)
            print((u''.join(output)).encode('utf8'))


if __name__ == '__main__':
    flags.mark_flag_as_required('input_file')
    app.run(main)

