# encoding: utf-8

import json

import tensorflow as tf
import os

from bert import modeling
from input_fns import file_based_input_fn_builder, CwsProcessor
from model_fns import model_fn_builder

flags = tf.flags

flags.DEFINE_string("train_data_path", default=None, help="")
flags.DEFINE_string("test_data_path", default=None, help="")
flags.DEFINE_string("label_dict_path", default=None, help="")

flags.DEFINE_integer("train_batch_size", default=16, help="")
flags.DEFINE_integer("eval_batch_size", default=16, help="")
flags.DEFINE_integer("predict_batch_size", default=16, help="")
flags.DEFINE_float("learning_rate", default=1e-5, help="")
flags.DEFINE_float("dropout", default=0.5, help="")
flags.DEFINE_integer("max_sequence_length", default=200, help="")

flags.DEFINE_string("ckpt_dir", default=None, help="")
flags.DEFINE_integer("save_checkpoints_steps", default=1000, help="")
flags.DEFINE_integer("log_step_count_steps", default=100, help="")
flags.DEFINE_integer("early_stopping_step", default=4,
                     help="Stop training after #early_stopping_step when loss not decrease")

# bert config
tf.app.flags.DEFINE_string("bert_config_file", default=None, help="")
tf.app.flags.DEFINE_string("vocab_file", default=None, help="")
tf.app.flags.DEFINE_string("init_checkpoint", default=None, help="")
tf.app.flags.DEFINE_bool("do_lower_case", default=True, help="")
# train parameters
flags.DEFINE_float("warmup_proportion", default=0.1, help="")
flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

# TPU config
flags.DEFINE_bool("use_tpu", default=False, help="")
flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

FLAGS = flags.FLAGS


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    processor = CwsProcessor(None)
    params = FLAGS.flag_values_dict()
    label_list = processor.get_labels()
    with open("{}.json".format(FLAGS.train_data_path)) as f:
        train_data_info = json.load(f)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.ckpt_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        log_step_count_steps=FLAGS.log_step_count_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    with open(os.path.join(FLAGS.ckpt_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    num_warmup_steps = int(train_data_info['num_train_steps'] * FLAGS.warmup_proportion)
    model_fn = model_fn_builder(
        bert_config,
        label_list,
        FLAGS.max_sequence_length,
        train_data_info['num_train_steps'],
        num_warmup_steps,
        FLAGS.init_checkpoint)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_input_fn = file_based_input_fn_builder(
            FLAGS.train_data_path,
            FLAGS.max_sequence_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=train_data_info['num_train_steps'])

    if FLAGS.do_eval:
        with open("{}.json".format(FLAGS.test_data_path)) as f:
            eval_data_info = json.load(f)
        num_eval_examples = eval_data_info['num_examples']

        if FLAGS.use_tpu:
            assert num_eval_examples % FLAGS.eval_batch_size == 0
            eval_steps = int(num_eval_examples // FLAGS.eval_batch_size)
        eval_input_fn = file_based_input_fn_builder(
            FLAGS.test_data_path,
            FLAGS.max_sequence_length,
            is_training=False,
            drop_remainder=False)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        output_eval_file = os.path.join(FLAGS.ckpt_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == '__main__':
    tf.app.run()
