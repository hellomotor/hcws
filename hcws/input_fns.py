# encoding: utf-8
import collections

import tensorflow as tf
import os
import codecs
from bert import tokenization
from absl import logging
import pickle
import re
from tqdm import tqdm
from utils import pfr_extract_tokens 

__all__ = ['DataProcessor', 'get_task_processor', 'CwsProcessor', 'NerProcessor', 'write_test_tokens',
           'convert_single_example', 'filed_based_convert_examples_to_features', 'file_based_input_fn_builder']


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid=None, text=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        raise NotImplementedError()


class CwsProcessor(DataProcessor):
    def __init__(self, output_dir):
        self.labels = set()
        self.output_dir = output_dir

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self, labels=None):
        return [
            "O", "a_B", "ad_B", "ad_I", "Ag_B", "a_I", "an_B", "an_I", "b_B",
            "Bg_B", "b_I", "c_B", "c_I", "d_B", "Dg_B", "d_I", "e_B", "e_I",
            "f_B", "f_I", "h_B", "i_B", "i_I", "j_B", "j_I", "k_B", "k_I",
            "l_B", "l_I", "m_B", "Mg_B", "m_I", "n_B", "Ng_B", "n_I", "nr_B",
            "nr_I", "ns_B", "ns_I", "nt_B", "nt_I", "nx_B", "nx_I", "nz_B",
            "nz_I", "o_B", "o_I", "p_B", "p_I", "q_B", "q_I", "r_B", "Rg_B",
            "r_I", "s_B", "s_I", "t_B", "Tg_B", "t_I", "u_B", "u_I", "v_B",
            "vd_B", "vd_I", "Vg_B", "v_I", "vn_B", "vn_I", "w_B", "w_I",
            "y_B", "Yg_B", "y_I", "z_B", "z_I", "X", "[CLS]", "[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    def _gen_token_labels(self, token, label):
        """ 
        :param token: 迈向
        :param label: v
        :return: [ [ 迈 向 ] [ v_B v_I ] ]
        """
        chars, labels = [], []
        for i, c in enumerate(token):
            chars.append(c)
            labels.append('{}_B'.format(label) if i == 0 else '{}_I'.format(label))
        return chars, labels

    def _read_data(self, input_file):
        """Reads a BIO data."""
        delimiter = u'\x002'
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            for i, line in enumerate(f):
                words, labels = [], []
                line = line.strip('\n')
                for text in pfr_extract_tokens(line):
                    i = text.rfind(u'/')
                    if i < 0:
                        continue
                    token, label = text[:i], text[i + 1:]
                    if not (token and label):
                        continue
                    chars, tags = self._gen_token_labels(token, label)
                    words.extend(chars)
                    labels.extend(tags)
                lines.append([' '.join(labels), ' '.join(words)])
                if line.startswith("-DOCSTART-"):
                    continue
            return lines


class NerProcessor(CwsProcessor):
    def get_labels(self, labels=None):
        """person, org, time, location, jobtitle"""
        return ["O", "na_B", "na_I", "person_B", "person_I", "org_B", "org_I", "time_B", "time_I",
                "location_B", "location_I", "jobtitle_B", "jobtitle_I", "X", "[CLS]", "[SEP]"]


def get_task_processor(name, *args):
    processors = {'cws': CwsProcessor, 'ner': NerProcessor}
    if name not in processors:
        raise ValueError('invalid task name, options are: [cws, ner]')
    return processors[name](*args)


def write_test_tokens(tokens, output_dir):
    """
    将序列解析结果写入到文件中
    只在mode=test的时候启用
    :param tokens:
    :param mode:
    :return:
    """
    path = os.path.join(output_dir, "token_test.txt")
    with codecs.open(path, 'a', encoding='utf-8') as wf:
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')


def convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer, output_dir, mode):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param output_dir
    :param mode:
    :return:
    """

    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for (word, label) in zip(textlist, labellist):
        # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），可以将所有的分字操作替换为list(input)
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i, _ in enumerate(token):
            if i == 0:
                labels.append(label)
            else:  # 一般不会出现else
                labels.append("X")
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    # 打印部分样本数据信息
    if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )
    # mode='test'的时候才有效
    if mode == 'test':
        write_test_tokens(ntokens, output_dir)
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, output_dir, mode=None):
    """
    将数据转化为TF_Record 结构，作为模型数据输入
    :param examples:  样本
    :param label_list:标签list
    :param max_seq_length: 预先设定的最大序列长度
    :param tokenizer: tokenizer 对象
    :param output_file: tf.record 输出路径
    :param mode:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    writer = tf.python_io.TFRecordWriter(output_file)
    # 遍历训练数据
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        # 对于每一个训练样本,
        feature = convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer, output_dir, mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # features["label_mask"] = create_int_feature(feature.label_mask)
        # tf.train.Example/Feature 是一种协议，方便序列化？？？
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=300)
        d = d.apply(tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_calls=8,  # 并行处理数据的CPU核心数量，不要大于你机器的核心数
            drop_remainder=drop_remainder))
        d = d.prefetch(buffer_size=4)
        return d

    return input_fn
