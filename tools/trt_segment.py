import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from tensorrtserver.api import *
import pickle
from absl import flags, app
from hcws.bert import tokenization
from hcws.utils import label_decode, batch_read_iter, load_plain_dict
import numpy as np
from functools import partial

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue


HOME = os.path.expanduser('~')

Q2B_DICT_FILE = os.path.join(HOME, 'git_repo/hcws/scripts/q2b.dic')
LABEL_DICT_FILE = os.path.join(HOME, 'git_repo/hcws/corpus/RMRB/train/label2id.pkl')
BERT_BASE_DIR = os.path.join(HOME, 'git_repo/tf_ner/bert_lstm_crf/chinese_L-12_H-768_A-12')
BERT_VOCAB = os.path.join(BERT_BASE_DIR, 'vocab.txt')

flags.DEFINE_string('url', 'localhost:8001', '')
flags.DEFINE_enum('protocol', 'grpc', enum_values=['http', 'grpc'], help='')
flags.DEFINE_bool('verbose', False, '')
flags.DEFINE_string('input_file', None, '')
flags.DEFINE_string('label_dict_file', LABEL_DICT_FILE, '')
flags.DEFINE_string('q2b_file', Q2B_DICT_FILE, '')
flags.DEFINE_string('vocab_file', BERT_VOCAB, '')
flags.DEFINE_integer('max_sequence_length', 200, '')
flags.DEFINE_integer('batch_size', 32, '')

FLAGS = flags.FLAGS


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def create_feature_from_tokens(tokenizer, q2b_dict, list_of_text, max_seq_length):
    # 保存label->index 的map
    batch_input_tokens, batch_input_ids, batch_input_mask, batch_segment_ids = [], [], [], []
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

        batch_input_tokens.append(ntokens)
        batch_input_ids.append(np.array(input_ids, dtype=np.int32))
        batch_input_mask.append(np.array(input_mask, dtype=np.int32))
        batch_segment_ids.append(np.array(segment_ids, dtype=np.int32))

    return {
        'input_tokens': batch_input_tokens,
        'input_ids': batch_input_ids,
        'input_mask': batch_input_mask,
        'segment_ids': batch_segment_ids,
    }


def completion_callback(id_to_label, batch_tokens, ctx, request_id):
    # user_data._completed_requests.put((ctx, request_id, batch_tokens))
    result = ctx.get_async_run_results(request_id, True)
    pred_ids = result['predictions']
    seq_lens = result['seq_lens']
    bsz = len(pred_ids)
    for i in range(bsz):
        end = seq_lens[i][0]  # [CLS] ... [SEP]
        label_ids = pred_ids[i][:end]
        labels = [id_to_label.get(_id, 'O') for _id in label_ids[1:-1]]
        tokens = batch_tokens[i][1:-1]
        output = label_decode(tokens, labels)
        print(''.join(output))


def async_segment():
    label_to_id = pickle.load(open(FLAGS.label_dict_file, 'rb'))
    id_to_label = dict([(v, k) for k, v in label_to_id.items()])
    q2b_dict = load_plain_dict(FLAGS.q2b_file)
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=True)

    protocol = ProtocolType.from_str(FLAGS.protocol)
    model_name = "hcws"
    model_version = -1
    ctx = InferContext(FLAGS.url, protocol, model_name, model_version, FLAGS.verbose)

    user_data = UserData()
    request_cnt = 0
    for idx, batch_of_line in enumerate(batch_read_iter(FLAGS.input_file, FLAGS.batch_size, from_line=0)):
        batch_of_tokens = [(w for w in line.strip('\n')[:FLAGS.max_sequence_length])
                           for line in batch_of_line]
        feature = create_feature_from_tokens(
                        tokenizer, 
                        q2b_dict, 
                        batch_of_tokens, 
                        FLAGS.max_sequence_length)
        result = ctx.async_run_with_cb(
            partial(completion_callback, id_to_label, feature['input_tokens']),
            { "input_ids": feature['input_ids'], "input_mask": feature['input_mask'], "segment_ids": feature['segment_ids'] },
            { "predictions": InferContext.ResultFormat.RAW, "seq_lens": InferContext.ResultFormat.RAW },
            batch_size=FLAGS.batch_size)
        request_cnt += 1


def sync_segment():
    label_to_id = pickle.load(open(FLAGS.label_dict_file, 'rb'))
    id_to_label = dict([(v, k) for k, v in label_to_id.items()])
    q2b_dict = load_plain_dict(FLAGS.q2b_file)
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=True)

    protocol = ProtocolType.from_str(FLAGS.protocol)
    model_name = "hcws"
    model_version = -1
    ctx = InferContext(FLAGS.url, protocol, model_name, model_version, FLAGS.verbose)

    for idx, batch_of_line in enumerate(batch_read_iter(FLAGS.input_file, FLAGS.batch_size, from_line=0)):
        batch_of_tokens = [(w for w in line.strip('\n')[:FLAGS.max_sequence_length])
                           for line in batch_of_line]
        feature = create_feature_from_tokens(
                        tokenizer, 
                        q2b_dict, 
                        batch_of_tokens, 
                        FLAGS.max_sequence_length)
        result = ctx.run(
            {  "input_ids": feature['input_ids'], "input_mask": feature['input_mask'], 
               "segment_ids": feature['segment_ids']},
            { "predictions": InferContext.ResultFormat.RAW, "seq_lens": InferContext.ResultFormat.RAW },
            batch_size=FLAGS.batch_size)
        pred_ids = result['predictions']
        seq_lens = result['seq_lens']
        input_tokens = feature['input_tokens']
        bsz = min(len(input_tokens), len(pred_ids))
        for i in range(bsz):
            end = seq_lens[i][0]  # [CLS] ... [SEP]
            label_ids = pred_ids[i][:end]
            labels = [id_to_label.get(_id, 'O') for _id in label_ids[1:-1]]
            tokens = input_tokens[i][1:-1]
            output = label_decode(tokens, labels)
            print(''.join(output))


def main(_):
    sync_segment()


if __name__ == '__main__':
    flags.mark_flag_as_required('input_file')
    app.run(main)
