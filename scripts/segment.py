# encoding: utf-8
import io
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import tensorflow as tf
from absl import flags, app
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util
from hcws.bert import tokenization
from hcws.utils import create_feature_from_tokens, label_decode

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
            output = label_decode(batch_of_text[i], labels)
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
