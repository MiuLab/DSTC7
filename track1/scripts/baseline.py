import argparse
import pdb
import sys
import traceback
import re
import json
from tqdm import tqdm


def main(args):
    # with open(args.train) as f:
    #     train = json.load(f)

    with open(args.valid) as f:
        valid = json.load(f)

    courses = []
    for sample in valid:
        courses += [c['offering'].split('-')[0].lower()
                    for c in (sample['profile']['Courses']['Prior']
                              + sample['profile']['Courses']['Suggested'])]

    courses = list(set(courses))

    n_correct, n_incorrect = 0, 0
    for sample in tqdm(valid):
        context_courses = set()
        for msg in sample['messages-so-far']:
            for w in msg['utterance'].split(' '):
                if w in courses:
                    context_courses.add(w)

        predict = ''
        max_intersect = -1
        for opt in sample['options-for-next']:
            opt_courses = set()
            for w in opt['utterance'].split(' '):
                if w in courses:
                    opt_courses.add(w)

            intersect = len(context_courses & opt_courses)
            if intersect > max_intersect:
                max_intersect = intersect
                predict = opt['utterance']

        if predict == [ans['utterance']
                       for ans in sample['options-for-correct-answers']][0]:
            n_correct += 1
        else:
            n_incorrect += 1

    print('accuracy = {}'.format(n_correct / len(valid)))
    assert (n_correct + n_incorrect) == len(valid)


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('train', type=str,
                        help='')
    parser.add_argument('valid', type=str,
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
