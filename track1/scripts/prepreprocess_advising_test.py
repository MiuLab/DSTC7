import argparse
import pdb
import sys
import traceback
import re
import json
from tqdm import tqdm
from prepreprocess_advising import collect_courses, ppprocess


def main(args):
    with open(args.train) as f:
        train = json.load(f)

    with open(args.valid) as f:
        valid = json.load(f)

    with open(args.test) as f:
        test = json.load(f)

    courses = collect_courses(train + valid)

    ppprocess(test, courses)
    with open(args.test_output, 'w') as f:
        json.dump(test, f, indent='    ')


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('train', type=str,
                        help='')
    parser.add_argument('valid', type=str,
                        help='')
    parser.add_argument('test', type=str,
                        help='')
    parser.add_argument('test_output', type=str,
                        help='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
