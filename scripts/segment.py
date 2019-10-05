# encoding: utf-8
"""Reload and serve a saved model"""

import io
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import numpy as np
import tensorflow as tf
from absl import flags, app
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util
from kcws.train.bert import tokenization


Q2B_DICT_FILE = '/home/heqing/git_repo/kcws/scripts/q2b.dic'
LABEL_DICT_FILE = '/home/heqing/git_repo/kcws/corpus/RMRB/train/label2id.pkl'
BERT_BASE_DIR = '/home/heqing/git_repo/tf_ner/bert_lstm_crf/chinese_L-12_H-768_A-12'
BERT_VOCAB = os.path.join(BERT_BASE_DIR, 'vocab.txt')

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', None, '')
flags.DEFINE_string('label_dict_file', LABEL_DICT_FILE, '')
flags.DEFINE_string('q2b_file', Q2B_DICT_FILE, '')
flags.DEFINE_string('vocab_file', BERT_VOCAB, '')
flags.DEFINE_integer('max_sequence_length', 200, '')
flags.DEFINE_integer('batch_size', 10, '')


def create_feature_from_tokens(tokenizer, q2b_dict, list_of_text, max_seq_length):
    # 保存label->index 的map
    batch_input_ids, batch_input_mask, batch_segment_ids = [], [], []
    for text in list_of_text:
        tokens = []
        for word in text:
            # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），可以将所有的分字操作替换为list(input)
            token = tokenizer.tokenize(q2b_dict.get(word, word))
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

        batch_input_ids.append(input_ids)
        batch_input_mask.append(input_mask)
        batch_segment_ids.append(segment_ids)

    return {
        'input_ids': np.array(batch_input_ids, dtype=np.int64),
        'input_mask': np.array(batch_input_mask, dtype=np.int64),
        'segment_ids': np.array(batch_segment_ids, dtype=np.int64),
    }


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


def create_grpc_stub():
    host, port = '192.168.199.120', 8500
    channel = grpc.insecure_channel('{}:{}'.format(host, port))
    return prediction_service_pb2_grpc.PredictionServiceStub(channel)


def inference(stub, features):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'saved_model'
    request.model_spec.signature_name = 'serving_default'
    for k in ('input_ids', 'input_mask', 'segment_ids'):
        request.inputs[k].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                features[k],
                shape=[FLAGS.batch_size,
                       FLAGS.max_sequence_length], dtype=tf.int32))
    return stub.Predict(request, 3.0)


def main(_):
    label_to_id = pickle.load(open(FLAGS.label_dict_file, 'rb'))
    id_to_label = dict([(v, k) for k, v in label_to_id.items()])
    q2b_dict = dict([line.strip('\n').split('\t') for line in io.open(FLAGS.q2b_file, encoding='utf8')])

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=True)
    batch_of_text, batch_of_length = [], []
    stub = create_grpc_stub()
    for line in io.open(FLAGS.input_file, encoding='utf8'):
        contents = line.strip('\n')
        words = [w for w in contents]
        if len(batch_of_text) < FLAGS.batch_size:
            batch_of_text.append(words)
            if len(words) >= FLAGS.max_sequence_length - 1:
                batch_of_length.append(FLAGS.max_sequence_length - 2)
            else:
                batch_of_length.append(len(words))
            continue
        feature = create_feature_from_tokens(tokenizer, q2b_dict, batch_of_text, FLAGS.max_sequence_length)
        result = inference(stub, feature)
        pred_ids = tensor_util.MakeNdarray(result.outputs['output'])
        for i in range(pred_ids.shape[0]):
            end = batch_of_length[i] + 2  # [CLS] ... [SEP]
            _ids = pred_ids[i][:end]
            labels = [id_to_label.get(_id, 'O') for _id in _ids[1:-1]]
            output = decode(batch_of_text[i], labels)
            print((u''.join(output)).encode('utf8'))
        batch_of_text[:], batch_of_length[:] = [], []


if __name__ == '__main__':
    flags.mark_flag_as_required('input_file')
    flags.mark_flag_as_required('label_dict_file')
    flags.mark_flag_as_required('vocab_file')
    try:
        app.run(main)
    except KeyboardInterrupt:
        print('User breaked .')
