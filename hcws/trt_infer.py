import io
import os
import tensorflow as tf
from absl import flags, app
from utils import create_feature_from_tokens, label_decode
import cPickle
from bert import tokenization
import tensorflow.contrib.tensorrt as trt

HOME = os.path.expanduser('~')

Q2B_DICT_FILE = os.path.join(HOME, 'git_repo/hcws/scripts/q2b.dic')
LABEL_DICT_FILE = os.path.join(HOME, 'git_repo/hcws/corpus/RMRB/train/label2id.pkl')
BERT_BASE_DIR = os.path.join(HOME, 'git_repo/tf_ner/bert_lstm_crf/chinese_L-12_H-768_A-12')
BERT_VOCAB = os.path.join(BERT_BASE_DIR, 'vocab.txt')

FLAGS = flags.FLAGS
flags.DEFINE_string('trt_pb_path', None, '')
flags.DEFINE_string('input_file', None, '')
flags.DEFINE_string('label_dict_file', LABEL_DICT_FILE, '')
flags.DEFINE_string('q2b_file', Q2B_DICT_FILE, '')
flags.DEFINE_string('vocab_file', BERT_VOCAB, '')
flags.DEFINE_integer('max_sequence_length', 200, '')
flags.DEFINE_integer('batch_size', 10, '')


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
    # load optimized graph_def to default graph
    graph = load_graph(trt_graph_def)
    # for op in graph.get_operations():
    #    sys.stderr.write(op.name + '\n')
    gpu_ops = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
    session_conf = tf.ConfigProto(allow_soft_placement=True,
                                  gpu_options=gpu_ops,
                                  inter_op_parallelism_threads=0,
                                  intra_op_parallelism_threads=0)
    sess = tf.Session(graph=graph, config=session_conf)
    t_preds = graph.get_tensor_by_name('ReverseSequence_1:0')
    t_sqls = graph.get_tensor_by_name('Sum:0')
    t_input_ids = graph.get_tensor_by_name('IteratorGetNext:0')
    t_input_mask = graph.get_tensor_by_name('IteratorGetNext:1')
    t_segment_ids = graph.get_tensor_by_name('IteratorGetNext:3')
    batch_of_text = []
    for line in io.open(FLAGS.input_file, encoding='utf8'):
        contents = line.strip('\n')[:FLAGS.max_sequence_length]
        words = [w for w in contents]
        if len(batch_of_text) < FLAGS.batch_size:
            batch_of_text.append(words)
            continue
        feature = create_feature_from_tokens(tokenizer, q2b_dict, batch_of_text, FLAGS.max_sequence_length)
        feed_dict = {t_input_ids: feature['input_ids'],
                     t_input_mask: feature['input_mask'],
                     t_segment_ids: feature['segment_ids']}
        pred_ids, seq_lens = sess.run([t_preds, t_sqls], feed_dict=feed_dict)
        for i in range(pred_ids.shape[0]):
            end = seq_lens[i] + 2  # [CLS] ... [SEP]
            _ids = pred_ids[i][:end]
            labels = [id_to_label.get(_id, 'O') for _id in _ids[1:-1]]
            output = label_decode(batch_of_text[i], labels)
            print((u''.join(output)).encode('utf8'))
        batch_of_text[:] = []


def load_graph(graph_def):
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            op_dict=None,
            producer_op_list=None,
            name='')

    return graph


def main(_):
    label_to_id = cPickle.load(open(FLAGS.label_dict_file, 'rb'))
    id_to_label = dict([(v, k) for k, v in label_to_id.items()])
    q2b_dict = dict([line.strip('\n').split('\t') for line in io.open(FLAGS.q2b_file, encoding='utf8')])

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=True)
    inference(FLAGS.trt_pb_path, tokenizer, q2b_dict, id_to_label)


if __name__ == '__main__':
    flags.mark_flag_as_required('trt_pb_path')
    flags.mark_flag_as_required('input_file')
    app.run(main)
