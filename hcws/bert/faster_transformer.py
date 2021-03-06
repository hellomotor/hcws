from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import device_lib
import os
import tensorflow as tf


build_path = '/home/heqing/git_repo/DeepLearningExamples/FasterTransformer/build/lib'
transformer_op_module = tf.load_op_library(
    os.path.join(build_path, 'libtf_fastertransformer.so'))


def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def fast_transformer_model_trans(input_tensor,
                                 attention_mask=None,
                                 hidden_size=768,
                                 num_hidden_layers=12,
                                 num_attention_heads=12,
                                 intermediate_size=3072,
                                 intermediate_act_fn=gelu,
                                 hidden_dropout_prob=0.1,
                                 attention_probs_dropout_prob=0.1,
                                 initializer_range=0.02,
                                 do_return_all_layers=False):
    """ Re-implementation of transformer_model function from modeling.py from Google's BERT repository https://github.com/google-research/bert
        using FasterTransformer Tensorflow op.
    Multi-headed, multi-layer Transformer from "Attention is All You Need".
    This is almost an exact implementation of the original Transformer encoder.
    See the original paper:
    https://arxiv.org/abs/1706.03762
    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
      attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
        seq_length], with 1 for positions that can be attended to and 0 in
        positions that should not be.
      hidden_size: int. Hidden size of the Transformer.
      num_hidden_layers: int. Number of layers (blocks) in the Transformer.
      num_attention_heads: int. Number of attention heads in the Transformer.
      intermediate_size: int. The size of the "intermediate" (a.k.a., feed
        forward) layer.
      intermediate_act_fn: function. The non-linear activation function to apply
        to the output of the intermediate/feed-forward layer.
      hidden_dropout_prob: float. Dropout probability for the hidden layers.
      attention_probs_dropout_prob: float. Dropout probability of the attention
        probabilities.
      initializer_range: float. Range of the initializer (stddev of truncated
        normal).
      do_return_all_layers: Whether to also return all layers or just the final
        layer.
    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size], the final
      hidden layer of the Transformer.
    Raises:
      ValueError: A Tensor shape or parameter is invalid.
    """
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                         (input_width, hidden_size))

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.
    prev_output = reshape_to_matrix(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output

            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    attention_head = attention_layer(
                        from_tensor=layer_input,
                        to_tensor=layer_input,
                        attention_mask=attention_mask,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length)
                    attention_heads.append(attention_head)

                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    # In the case where we have other sequences, we just concatenate
                    # them to the self-attention head before the projection.
                    attention_output = tf.concat(attention_heads, axis=-1)

                # Run a linear projection of `hidden_size` then add a residual
                # with `layer_input`.
                with tf.variable_scope("output"):
                    attention_output = tf.layers.dense(
                        attention_output,
                        hidden_size,
                        kernel_initializer=create_initializer(initializer_range))
                    attention_output = dropout(
                        attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(
                        attention_output + layer_input)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(
                    attention_output,
                    intermediate_size,
                    activation=intermediate_act_fn,
                    kernel_initializer=create_initializer(initializer_range))

            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(
                    intermediate_output,
                    hidden_size,
                    kernel_initializer=create_initializer(initializer_range))
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)

            # FASTINFER: fast transformer encoder inference
            trainable_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
            layer_output = transformer_op_module.bert_transformer(
                layer_input,
                layer_input,
                trainable_vars[0], trainable_vars[2], trainable_vars[4], trainable_vars[1], trainable_vars[3], trainable_vars[5],
                attention_mask,
                trainable_vars[6], trainable_vars[7], trainable_vars[8], trainable_vars[9], trainable_vars[10], trainable_vars[11],
                trainable_vars[12], trainable_vars[13], trainable_vars[14], trainable_vars[15],
                batch_size=batch_size, from_seq_len=seq_length, to_seq_len=seq_length, head_num=num_attention_heads, size_per_head=attention_head_size)

            prev_output = layer_output
            all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output

