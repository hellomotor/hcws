from absl import flags, app
import ahocorasick
import pickle
import json

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', None, '')


def main(_):
    job_titles = set()
    with open(FLAGS.input_file) as f:
        for line in f:
            d = json.loads(line.strip())
            text, labels = d['text'], d['labels']
            if not labels:
                continue
            for label in labels:
                start, end, name = label
                if name == 'JOB_TITLE':
                    job_titles.add(text[start:end])
    A = ahocorasick.Automaton()
    for jt in job_titles:
        A.add_word(jt, (jt))
    A.make_automaton()
    with open('./job_titles.trie', 'wb') as f:
        pickle.dump(A, f)


if __name__ == '__main__':
    flags.mark_flag_as_required('input_file')
    app.run(main)
