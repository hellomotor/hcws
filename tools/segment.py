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

HOME = os.path.expanduser('~')

Q2B_DICT_FILE = os.path.join(HOME, 'git_repo/hcws/scripts/q2b.dic')
LABEL_DICT_FILE = os.path.join(HOME, 'git_repo/hcws/corpus/RMRB/train/label2id.pkl')
BERT_BASE_DIR = os.path.join(HOME, 'git_repo/tf_ner/bert_lstm_crf/chinese_L-12_H-768_A-12')
BERT_VOCAB = os.path.join(BERT_BASE_DIR, 'vocab.txt')

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', None, '')
flags.DEFINE_string('label_dict_file', LABEL_DICT_FILE, '')
flags.DEFINE_string('q2b_file', Q2B_DICT_FILE, '')
flags.DEFINE_string('model_name', None, '')
flags.DEFINE_string('vocab_file', BERT_VOCAB, '')
flags.DEFINE_integer('max_sequence_length', 200, '')
flags.DEFINE_integer('batch_size', 10, '')


def create_grpc_stub():
    host, port = '192.168.199.120', 8500
    channel = grpc.insecure_channel('{}:{}'.format(host, port))
    return prediction_service_pb2_grpc.PredictionServiceStub(channel)


def inference(stub, features):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = FLAGS.model_name
    request.model_spec.signature_name = 'serving_default'
    for k in ('input_ids', 'input_mask', 'segment_ids'):
        request.inputs[k].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                features[k],
                shape=[FLAGS.batch_size,
                       FLAGS.max_sequence_length], dtype=tf.int32))
    return stub.Predict(request, 3.0)


def predict(stub, tokenizer, q2b_dict, id_to_label, batch_of_text):
    feature = create_feature_from_tokens(tokenizer, q2b_dict, batch_of_text, FLAGS.max_sequence_length)
    result = inference(stub, feature)
    seq_lens = tensor_util.MakeNdarray(result.outputs['seq_lens'])
    pred_ids = tensor_util.MakeNdarray(result.outputs['predictions'])
    input_ids = feature['input_ids']
    for i in range(pred_ids.shape[0]):
        end = seq_lens[i]  # [CLS] ... [SEP]
        label_ids = pred_ids[i][:end]
        token_ids = input_ids[i][:end]
        labels = [id_to_label.get(_id, 'O') for _id in label_ids[1:-1]]
        tokens = tokenizer.convert_ids_to_tokens(token_ids[1:-1])
        output = label_decode(tokens, labels)
        print((u''.join(output)).encode('utf8'))


def main(_):
    label_to_id = pickle.load(open(FLAGS.label_dict_file, 'rb'))
    id_to_label = dict([(v, k) for k, v in label_to_id.items()])
    q2b_dict = dict([line.strip('\n').split('\t') for line in io.open(FLAGS.q2b_file, encoding='utf8')])

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=True)
    batch_of_text = []
    stub = create_grpc_stub()
    for line in io.open(FLAGS.input_file, encoding='utf8'):
        contents = line.strip('\n')[:FLAGS.max_sequence_length]
        words = [w for w in contents]
        if len(batch_of_text) < FLAGS.batch_size:
            batch_of_text.append(words)
            continue
        predict(stub, tokenizer, q2b_dict, id_to_label, batch_of_text)
        batch_of_text[:] = []
    if batch_of_text:
        predict(stub, tokenizer, q2b_dict, id_to_label, batch_of_text)
        batch_of_text[:] = []


if __name__ == '__main__':
    flags.mark_flag_as_required('input_file')
    flags.mark_flag_as_required('label_dict_file')
    flags.mark_flag_as_required('vocab_file')
    flags.mark_flag_as_required('model_name')
    try:
        app.run(main)
    except KeyboardInterrupt:
        print('User breaked .')
