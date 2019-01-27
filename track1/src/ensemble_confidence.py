import argparse
import pdb
import sys
import traceback
import logging
import json


def main(args):
    predicts = []
    for path in args.inputs:
        with open(path) as f:
            predicts.append(json.load(f))

    ensembled = []
    for examples in zip(*predicts):
        example_id = examples[0]['example-id']
        n_candidates = len(examples[0]['candidate-ranking'])

        rankings = []
        for example in examples:
            assert len(example['candidate-ranking']) == n_candidates
            assert example['example-id'] == example_id
            rankings.append(
                sorted(example['candidate-ranking'],
                       key=lambda a: str(a['candidate-id']))
            )

        ensembled_rankings = []
        for candidates in zip(*rankings):
            candidate_id = candidates[0]['candidate-id']
            confident_score = 0
            for candidate in candidates:
                assert candidate['candidate-id'] == candidate_id
                confident_score += candidate['confidence']

            confident_score /= len(candidates)
            ensembled_rankings.append({
                'candidate-id': candidate_id,
                'confidence': confident_score
            })
        ensembled_rankings = sorted(ensembled_rankings,
                                    key=lambda a: a['confidence'],
                                    reverse=True)

        ensembled.append({
            'example-id': example_id,
            'candidate-ranking': ensembled_rankings
        })

    with open(args.output, 'w') as f:
        json.dump(ensembled, f, indent='  ')


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('inputs', type=str, nargs='+',
                        help='One or more prediction file paths.'
                        '(Accept arbitrary number of args.)')
    parser.add_argument('output', type=str,
                        help='Path to the destination.')
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
