# encoding: utf-8
import numpy as np


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


def label_decode(tokens, labels):
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
