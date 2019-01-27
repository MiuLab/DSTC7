import argparse
import pdb
import pickle
import sys
import traceback
import numpy as np


def main(args):
    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    first_utt_len = np.array([sample['utterance_ends'][0]
                             for sample in dataset])
    total_len = np.array([sample['utterance_ends'][-1]
                          for sample in dataset])
    dialog_len = np.array([len(sample['utterance_ends'])
                           for sample in dataset])
    utterance_len_x1 = np.array(
        [sample['utterance_ends'][i] - sample['utterance_ends'][i - 1]
         for sample in dataset
         for i in range(1, len(sample['utterance_ends']))
        ])

    print('| First Utterance Len Mean / Std. '
          '| Dialog Len Mean / Std. '
          '| Utterance Len Mean / Std. '
          # '| Utterance Len (exclude first) Mean / Std. '
          '| Total Len Mean / Std.| \n'
          '| --- | --- | --- | --- | \n'
          '| {first_utt_len:.4f} / {first_utt_len_std:.4f} '
          '| {dialog_len:.4f} / {dialog_len_std:.4f} '
          # '| {utt_len:.4f} / {utt_len_std:.4f} '
          '| {utt_len_x1:.4f} / {utt_len_x1_std:.4f} '
          '| {total_len:.4f} / {total_len_std:.4f} |'.format(
              first_utt_len=np.mean(first_utt_len),
              first_utt_len_std=np.std(first_utt_len),
              dialog_len=np.mean(dialog_len),
              dialog_len_std=np.std(dialog_len),
              # utt_len=np.mean(utterance_len),
              # utt_len_std=np.std(utterance_len),
              utt_len_x1=np.mean(utterance_len_x1),
              utt_len_x1_std=np.std(utterance_len_x1),
              total_len=np.mean(total_len),
              total_len_std=np.std(total_len))
    )

def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('dataset_path', type=str,
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
