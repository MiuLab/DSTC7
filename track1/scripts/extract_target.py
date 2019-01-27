import argparse
import pdb
import sys
import traceback
import logging
import json


def main(args):
    with open(args.data_path) as f:
        data = json.load(f)

    with open(args.output_path, 'w') as f:

        for sample in data:
            if len(sample['options-for-correct-answers']) > 0:
                candidates = ','.join(
                    [str(opt['candidate-id'])
                     for opt in sample['options-for-correct-answers']]
                )
            else:
                candidates = 'NONE'

            f.write(
                '{}\t{}\n'.format(sample['example-id'], candidates)
            )


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('data_path', type=str,
                        help='')
    parser.add_argument('output_path', type=str,
                        help='')

    args = parser.parse_args()  
    return args


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
