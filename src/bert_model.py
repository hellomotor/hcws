import tensorflow as tf
from kcws.train.bert import modeling
import sys
sys.path.append('/home/heqing/git_repo/kcws/ops')


class BertModel:
    def __init__(self, hps):
        self.hps = hps
        self.words = tf.Variable(self.c2v, name="words")
        self.inp = tf.placeholder(tf.int32,
                                  shape=[None, None],
                                  name="input_placeholder")

        self.bert_config = modeling.BertConfig.from_json_file(hps.bert_config_file)

    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def encode(self, X, length, reuse=False):
        input_ids, input_mask, segment_ids, labels = X
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=False if reuse else True,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)
        embedding = model.get_sequence_output()
        output = tf.layers.dropout(embedding, rate=self.hps.dropout, training=not reuse)
        output_dim = output.shape[-1].value
        self.W = tf.get_variable(
            shape=[output_dim, self.hps.num_tags],
            initializer=tf.contrib.layers.xavier_initializer(),
            name="weights",
            regularizer=tf.contrib.layers.l2_regularizer(0.001))
        self.b = tf.Variable(tf.zeros([self.hps.num_tags], name="bias"))
        matricized_unary_scores = tf.nn.xw_plus_b(output, self.W, self.b)
        unary_scores = tf.reshape(
            matricized_unary_scores,
            [-1, tf.shape(input_ids)[1], self.hps.num_tags],
            name="Reshape_7" if reuse else None)
        return unary_scores

    def inference(self, X, reuse=None):
        length = self.length(X)
        unary_scores = self.encode(X, length, reuse=reuse)
        return unary_scores, length

    def loss(self, X):
        _, _, _, Y = X
        P, sequence_length = self.inference(X)
        iters = tf.shape(X)[0] * tf.shape(X)[1]
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            P, Y, sequence_length)
        loss = tf.reduce_mean(-log_likelihood)
        return loss, iters

    def test_unary_score(self):
        P, sequence_length = self.inference(self.inp, reuse=True)
        pred_seqs, _ = tf.contrib.crf.crf_decode(P, self.transition_params, sequence_length)
        pred_seqs = tf.identity(pred_seqs, name='pred_seqs')
        return P, sequence_length
