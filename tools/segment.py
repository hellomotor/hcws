# encoding: utf-8
import io
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import tensorflow as tf
from absl import flags, app, logging
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util
from hcws.bert import tokenization
from hcws.utils import create_feature_from_tokens, label_decode, batch_read_iter, load_plain_dict

HOME = os.path.expanduser('~')

Q2B_DICT_FILE = os.path.join(HOME, 'git_repo/hcws/scripts/q2b.dic')
LABEL_DICT_FILE = os.path.join(HOME, 'git_repo/hcws/corpus/RMRB/train/label2id.pkl')
BERT_BASE_DIR = os.path.join(HOME, 'git_repo/tf_ner/bert_lstm_crf/chinese_L-12_H-768_A-12')
BERT_VOCAB = os.path.join(BERT_BASE_DIR, 'vocab.txt')

FLAGS = flags.FLAGS
flags.DEFINE_string('host', '127.0.0.1', '')
flags.DEFINE_integer('port', 8500, '')
flags.DEFINE_string('input_file', None, '')
flags.DEFINE_string('label_dict_file', LABEL_DICT_FILE, '')
flags.DEFINE_string('q2b_file', Q2B_DICT_FILE, '')
flags.DEFINE_string('model_name', None, '')
flags.DEFINE_string('vocab_file', BERT_VOCAB, '')
flags.DEFINE_integer('max_sequence_length', 200, '')
flags.DEFINE_integer('batch_size', 10, '')
flags.DEFINE_integer('from_line', 0, '')
flags.DEFINE_bool('ignore_space', False, 'ignore space chars in input_file.')
flags.DEFINE_bool('output_pos', True, 'output POS tag.')


def create_grpc_stub():
    channel = grpc.insecure_channel('{}:{}'.format(FLAGS.host, FLAGS.port))
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
    feature = create_feature_from_tokens(
        tokenizer, 
        q2b_dict, 
        batch_of_text, 
        FLAGS.max_sequence_length)
    result = inference(stub, feature)
    seq_lens = tensor_util.MakeNdarray(result.outputs['seq_lens'])
    pred_ids = tensor_util.MakeNdarray(result.outputs['predictions'])
    input_ids = feature['input_ids']
    batch_size = min(len(batch_of_text), pred_ids.shape[0])
    for i in range(batch_size):
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
    q2b_dict = load_plain_dict(FLAGS.q2b_file)

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=True)
    try:
        stub = create_grpc_stub()
        if FLAGS.from_line > 0:
            logging.info('*' * 60)
            logging.info('file {}, start from line {}'.format(FLAGS.input_file, FLAGS.from_line))
            logging.info('*' * 60)
        for batch_of_line in batch_read_iter(FLAGS.input_file, FLAGS.batch_size, from_line=FLAGS.from_line):
            batch_of_tokens = [(w for w in line.strip('\n')[:FLAGS.max_sequence_length])
                               for line in batch_of_line]
            predict(stub, tokenizer, q2b_dict, id_to_label, batch_of_tokens)
    except KeyboardInterrupt:
        print('User breaked .')


if __name__ == '__main__':
    flags.mark_flag_as_required('input_file')
    flags.mark_flag_as_required('model_name')
    app.run(main)
