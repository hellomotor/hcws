from absl import flags, app
import json

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', None, '')

LABEL_MAP = {'JOB_TITLE': 'jobtitle',
             'LOCATION': 'location',
             'ORG_NAME': 'org',
             'PERSON_NAME': 'person',
             'TIME': 'time'}


def main(_):
    NAE = 'O'
    with open(FLAGS.input_file) as f:
        for line in f:
            d = json.loads(line.strip())
            text, labels = d['text'], d['labels']
            if not labels:
                continue
            chars = list(text)
            output = []
            last_idx = 0
            for label in sorted(labels, key=lambda l: l[0]):
                start, end, name = label
                name = LABEL_MAP.get(name, name)
                for i in range(last_idx, start):
                    output.append('{}/{}'.format(chars[i], NAE))
                output.append('{}/{}'.format(''.join(chars[start:end]), name))
                last_idx = end
            for i in range(last_idx, len(chars)):
                output.append('{}/{}'.format(chars[i], NAE))
            print(' '.join(output))


if __name__ == '__main__':
    flags.mark_flag_as_required('input_file')
    app.run(main)
