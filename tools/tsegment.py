import os
import tensorflow as tf
from absl import flags, app
from hcws.utils import create_feature_from_tokens, label_decode, batch_read_iter, load_plain_dict
import cPickle
from hcws.bert import tokenization
import tensorflow.contrib.tensorrt as trt

HOME = os.path.expanduser('~')

Q2B_DICT_FILE = os.path.join(HOME, 'git_repo/hcws/scripts/q2b.dic')
LABEL_DICT_FILE = os.path.join(HOME, 'git_repo/hcws/corpus/RMRB/train/label2id.pkl')
BERT_BASE_DIR = os.path.join(HOME, 'git_repo/tf_ner/bert_lstm_crf/chinese_L-12_H-768_A-12')
BERT_VOCAB = os.path.join(BERT_BASE_DIR, 'vocab.txt')

INPUT_IDS, INPUT_MASK, SEGMENT_IDS = 'IteratorGetNext:0', 'IteratorGetNext:1', 'IteratorGetNext:3'
OUTPUT_PREDS, OUTPUT_SQLS = 'pred_ids:0', 'seq_lengths:0'

FLAGS = flags.FLAGS
flags.DEFINE_string('trt_pb_path', None, '')
flags.DEFINE_string('input_file', None, '')
flags.DEFINE_string('label_dict_file', LABEL_DICT_FILE, '')
flags.DEFINE_string('q2b_file', Q2B_DICT_FILE, '')
flags.DEFINE_string('vocab_file', BERT_VOCAB, '')
flags.DEFINE_integer('max_sequence_length', 200, '')
flags.DEFINE_integer('batch_size', 16, '')


def load_frozen_graph_def(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def


def inference(trt_pb_path, tokenizer, q2b_dict, id_to_label):
    with tf.gfile.GFile(trt_pb_path, "rb") as f:
        trt_graph_def = tf.GraphDef()
        trt_graph_def.ParseFromString(f.read())

    tf.reset_default_graph()
    return_elements = [INPUT_IDS, INPUT_MASK, SEGMENT_IDS, OUTPUT_PREDS, OUTPUT_SQLS]
    # load optimized graph_def to default graph
    with tf.Graph().as_default() as graph:
        input_output_tensors = tf.import_graph_def(
            trt_graph_def,
            input_map=None,
            return_elements=return_elements)
    gpu_ops = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
    session_conf = tf.ConfigProto(allow_soft_placement=True,
                                  gpu_options=gpu_ops,
                                  inter_op_parallelism_threads=0,
                                  intra_op_parallelism_threads=0)
    sess = tf.Session(graph=graph, config=session_conf)
    for batch_of_line in batch_read_iter(FLAGS.input_file, FLAGS.batch_size):
        batch_of_tokens = [(w for w in line.strip('\n')[:FLAGS.max_sequence_length])
                           for line in batch_of_line]
        predict(sess, batch_of_tokens, tokenizer, q2b_dict, input_output_tensors, id_to_label)


def predict(session, batch_of_text, tokenizer, q2b_dict, input_output_tensors, id_to_label):
    (t_input_ids, t_input_mask, t_segment_ids, t_preds, t_sqls) = input_output_tensors
    feature = create_feature_from_tokens(tokenizer, q2b_dict, batch_of_text, FLAGS.max_sequence_length)
    feed_dict = {t_input_ids: feature['input_ids'],
                 t_input_mask: feature['input_mask'],
                 t_segment_ids: feature['segment_ids']}
    pred_ids, seq_lens, input_ids = session.run(
        [t_preds, t_sqls, t_input_ids], feed_dict=feed_dict)
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
    label_to_id = cPickle.load(open(FLAGS.label_dict_file, 'rb'))
    id_to_label = dict([(v, k) for k, v in label_to_id.items()])
    q2b_dict = load_plain_dict(FLAGS.q2b_file)

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=True)
    inference(FLAGS.trt_pb_path, tokenizer, q2b_dict, id_to_label)


if __name__ == '__main__':
    flags.mark_flag_as_required('trt_pb_path')
    flags.mark_flag_as_required('input_file')
    app.run(main)
