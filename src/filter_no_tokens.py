import argparse
import json


def main(_):
    with open(FLAGS.input, 'r') as f:
        for line in f:
            data = json.loads(line)
            if len(data['tokens']) > 0:
                print(json.dumps(data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    FLAGS, _ = parser.parse_known_args()

    main(_)

