import os
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', None, '')
flags.DEFINE_string('model', None, '')
flags.DEFINE_integer('batch_size', 16, '')

TRT_SUFFIX = '_fp32_trt'
output_names = ['ReverseSequence_1:0', 'Sum:0']


def main(_):
    print("------------- Load frozen graph from disk -------------")
    with tf.gfile.GFile(os.path.join(FLAGS.data_dir, FLAGS.model + '.pb'), "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    print("------------- Optimize the model with TensorRT -------------")
    trt_graph = trt.create_inference_graph(
        input_graph_def=graph_def,
        outputs=output_names,
        max_batch_size=FLAGS.batch_size,
        max_workspace_size_bytes=(1 << 30) * 4,
        precision_mode='FP32',
        minimum_segment_size=10
    )
    print("------------- Write optimized model to the file -------------")
    with open(os.path.join(FLAGS.data_dir, FLAGS.model + TRT_SUFFIX + '.pb'), 'wb') as f:
        f.write(trt_graph.SerializeToString())
    print("------------- DONE! -------------")


if __name__ == '__main__':
    flags.mark_flag_as_required('data_dir')
    flags.mark_flag_as_required('model')
    app.run(main)
