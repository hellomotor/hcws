import tensorflow as tf
from bert import modeling, optimization
from absl import logging


def model_fn_builder(bert_config,
                     label_list,
                     max_sequence_length,
                     num_train_steps,
                     num_warmup_steps,
                     init_checkpoint,
                     use_crf=True):
    def model_fn(features, labels, mode, params):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        dropout = params['dropout']
        learning_rate = params['learning_rate']
        num_tags = len(label_list) + 1

        input_ids = features['input_ids']
        input_mask = features['input_mask']
        segment_ids = features['segment_ids']
        if "label_ids" in features.keys():
            label_ids = features["label_ids"]
        else:
            label_ids = None

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)
        embedding = model.get_sequence_output()
        output = tf.layers.dropout(embedding, rate=dropout, training=is_training)
        logits = tf.layers.dense(
            output,
            num_tags,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))

        used = tf.sign(tf.abs(input_ids))
        seq_lengths = tf.cast(tf.reduce_sum(used, reduction_indices=1), tf.int32)

        if use_crf:
            # CRF
            crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
            pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, seq_lengths)
            if label_ids is not None:
                log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                    logits, label_ids, seq_lengths, crf_params)
                loss = tf.reduce_mean(-log_likelihood)
        else:
            probabilities = tf.nn.softmax(logits, axis=-1)
            pred_ids = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
            if label_ids is not None:
                cross_entropy = label_ids * tf.log(tf.nn.softmax(logits))
                cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
                cross_entropy *= tf.to_float(used)
                cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
                cross_entropy /= tf.cast(seq_lengths, tf.float32)
                loss = tf.reduce_mean(cross_entropy)

        logging.info('*' * 120)
        logging.info('input_ids: {}, input_mask: {}, segment_ids: {}'.format(input_ids, input_mask, segment_ids))
        logging.info('-' * 120)
        logging.info('seq_lengths: {}, pred_ids: {}'.format(seq_lengths, pred_ids))
        logging.info('*' * 120)

        tvars = tf.trainable_variables()
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, False)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            weights = tf.sequence_mask(seq_lengths, max_sequence_length)

            def metric_fn(label_ids, pred_ids):
                return {
                    'accuracy': tf.metrics.accuracy(label_ids, pred_ids, weights=weights)
                }

            eval_metrics = metric_fn(label_ids, pred_ids)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metrics)
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    "seq_lens": seq_lengths,
                    "predictions": pred_ids
                })
        return output_spec

    return model_fn
