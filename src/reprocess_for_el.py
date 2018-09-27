"""Changes the way sentences are annotated."""
import argparse
import json


def reflow(data):
    tokens = data['tokens']
    tokens = [x.strip() for x in tokens if x.strip!='']
    sentences = data['sentences']
    reflowed = []
    for start, end in zip(sentences[:-1], sentences[1:]):
        sentence = tokens[start:end]
        reflowed.append(sentence)
    out = data.copy()
    del out['sentences']
    out['tokens'] = reflowed
    return out


def main(_):
    for fname in FLAGS.inputs:
        with open(fname, 'r') as f, open(fname+'.tmp', 'w') as g:
            for line in f:
                data = json.loads(line)
                out = reflow(data)
                if 'tokens' in out:
                    if out['tokens'] != []:
                        g.write(json.dumps(out)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', type=str)
    FLAGS, _ = parser.parse_known_args()

    main(_)

