import numpy as np
import os
from absl import flags, app


flags.DEFINE_string('input_file', default=None, help='')
flags.DEFINE_float('test_ratio', default=0.01, help='')
flags.DEFINE_float('eval_ratio', default=0.02, help='')
FLAGS = flags.FLAGS


def main(_):
    cwd = '.'

    train_file_path = os.path.join(cwd, 'train.txt')
    eval_file_path = os.path.join(cwd, 'dev.txt')
    test_file_path = os.path.join(cwd, 'test.txt')
    print('*' * 120)
    print('input filename: {}'.format(FLAGS.input_file))
    print('\ttest set ratio: {}'.format(FLAGS.test_ratio))
    print('\tdev  set ratio: {}'.format(FLAGS.eval_ratio))
    print('\ttrain set filename: {}'.format(train_file_path))
    print('\teval set filename: {}'.format(eval_file_path))
    print('\ttest set filename: {}'.format(test_file_path))
    print('*' * 120)

    with open(str(train_file_path), 'w') as f_train, open(eval_file_path, 'w') as f_dev, open(test_file_path, 'w') as f_test:
        for line in open(FLAGS.input_file, 'r'):
            r = np.random.rand()
            if r < FLAGS.test_ratio:
                f_test.write(line)
            elif FLAGS.test_ratio < r < FLAGS.eval_ratio:
                f_dev.write(line)
            else:
                f_train.write(line)


if __name__ == '__main__':
    flags.mark_flag_as_required('input_file')
    app.run(main)
