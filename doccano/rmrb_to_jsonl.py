from absl import flags, app
import json
import ahocorasick

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', None, '')
flags.DEFINE_string('dict_file', None, '')


def main(_):
    with open(FLAGS.dict_file, 'rb') as f
        jobtitles = pickle.load(f)

    label_map = {'nr': 'PERSON_NAME', 'nt': 'ORG_NAME', 'ns': 'LOCATION', 't': 'TIME'}
    with open(FLAGS.input_file) as f:
        for line in f:
            tokens, labels = [], []
            start = 0
            prev_label = None
            for token in line.strip().split(' '):
                i = token.rfind('/')
                tok, pos = token[:i], token[i+1:]
                if (prev_label == 't' and pos == 't') or (prev_label == 'ns' and pos == 'ns'):
                    if tokens:
                        tokens[-1] += tok
                        p_start, p_end, p_pos = labels[-1]
                        labels[-1] = [p_start, p_end + len(tok), p_pos]
                        start = p_end + len(tok)
                    else:
                        labels.append([start, start + len(tok), labem_map[pos]])
                        tokens.append(tok)
                        start += len(tok)
                else:
                    tokens.append(tok)
                    if pos in ('nr', 'nt', 'ns', 't'):
                        labels.append([start, start + len(tok), label_map[pos]])
                    if tok == '记者':
                        labels.append([start, start + len(tok), 'JOB_TITLE'])
                    start += len(tok)
                prev_label = pos
            text = ''.join(tokens)
            for end_index, token in jobtitles.iter(text):
                start_index = end_index - len(token) + 1
                end_index = start_index + len(token)

            # print(text)
            # for (s, e, t) in labels:
            #     print('\t%s:\t%s' % (t, text[s:e]))
            # continue
            output = {'text': text, "labels": labels}
            has_org, has_person = False, False
            a = json.dumps(output, ensure_ascii=False)
            for (s, e, t) in labels:
                if t == 'ORG_NAME': has_org = True
                if t == 'PERSON_NAME': has_person = True
                # print('\t%s:%s' % (t, text[s:e]))
            if has_org and has_person:
                print(a)
            

if __name__ == '__main__':
    flags.mark_flag_as_required('input_file')
    app.run(main)
