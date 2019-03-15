from src.model.model import DialogModel
from src.log import log
from src.dataset.dataset import AVSD_Dataset
from src.util import utils
from src.util.WordHelper import WordHelper

import argparse
import torch
from pathlib import Path

def init(args):
    # Generate timestamp
    if args.parameter_timestamp is not None:
        timestamp = utils.generate_timestamp()
        # Restore training
        timestamp = args.parameter_timestamp
        restore_training = True
    else:
        # Generate timestamp
        timestamp = utils.generate_timestamp()
        restore_training = False

    log.logging.debug("Timestamp: {}".format(timestamp))
    # Load WordHelper
    word_helper = WordHelper(args.data_base_path / "word_embedding" / "glove.6B.300d.txt", "glove", 
                                cacheDir=args.data_base_path / "cache", 
                                timestamp=timestamp, 
                                word_embed_size=300, 
                                requires_grad=args.finetune,
                                elmo=args.elmo)
    # build dataset
    train_dataset = AVSD_Dataset(args.data_base_path, timestamp, word_helper, data_category="train", video_upsample=args.video_upsample)
    valid_dataset = AVSD_Dataset(args.data_base_path, timestamp, word_helper, data_category="valid", video_upsample=args.video_upsample)
    # model 
    model = DialogModel(train_dataset, 
                        args.batch_size, 
                        args.shuffle, 
                        args.outputDir, 
                        args.modelType, 
                        word_helper,
                        timestamp,
                        args.context,
                        restore_training)
    model.build()
    
    # cuda
    if args.cuda:
        model.cuda()
    return model, train_dataset, valid_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_base_path", type=Path, default="./data")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--shuffle", type=utils.str2bool, default="True")
    parser.add_argument("--cuda", type=utils.str2bool, default="True")
    parser.add_argument("--outputDir", type=Path, default="./output/")
    parser.add_argument("--modelType", type=str, default="SimpleModel")
    parser.add_argument("--context", type=int, default=1, help='use whole dialog as context or others')
    parser.add_argument('--elmo', type=int, default=0)
    parser.add_argument('--video_upsample', action='store_true', default=False)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument("--parameter_timestamp", type=str, default=None)
    args = parser.parse_args()
    # Init
    model, train_dataset, valid_dataset = init(args)
    # 
    log.logging.debug("[*] Model Fitting")
    model.fit(train_dataset, valid_dataset)
    log.logging.debug("[-] Model Done Fitting")
    # Save model
    #model.cpu()
    #model.save() 

if __name__ == "__main__":
    main()