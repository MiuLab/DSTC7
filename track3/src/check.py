from AVSD_Jim.src.model.model import DialogModel
from AVSD_Jim.src.log import log
from AVSD_Jim.src.dataset.dataset import AVSD_Dataset
from AVSD_Jim.src.util import utils
from AVSD_Jim.src.util.WordHelper import WordHelper

import argparse
import torch
from pathlib import Path

def init(args):
    # Generate timestamp
    timestamp = utils.generate_timestamp()
    log.logging.debug("Timestamp: {}".format(timestamp))
    # Load WordHelper
    word_helper = WordHelper(args.data_base_path / "word_embedding" / "glove.6B.300d.txt", "glove", 
                                cacheDir=args.data_base_path / "cache", 
                                timestamp=timestamp, 
                                word_embed_size=300, 
                                requires_grad=False)
    # build dataset
    train_dataset = AVSD_Dataset(args.data_base_path, timestamp, word_helper, data_category="train")
    valid_dataset = AVSD_Dataset(args.data_base_path, timestamp, word_helper, data_category="valid")
    test_dataset = AVSD_Dataset(args.data_base_path, timestamp, word_helper, data_category='test')
    
    return train_dataset, valid_dataset, test_dataset, word_helper

def check_data(dataset):
    i3d_flow_len = []
    i3d_rgb_len = []
    for (i3d_flow, i3d_rgb, 
        vggish, caption, caption_seq_len, question_list, questions_seq_len, answer_list, answers_seq_len) in dataset:
        i3d_flow_len.append(i3d_flow.shape[0])
        i3d_rgb_len.append(i3d_rgb.shape[0])
    print(sorted(i3d_flow_len))
    print(sorted(i3d_rgb_len))


def check_dataset(dataset, word_helper):
    data_loader = dataset.batch_generator(32, 
                                        shuffle=True, 
                                        word_helper=word_helper, 
                                        use_cuda=False, 
                                        training=False)
    i3d_flow_len = 1000
    i3d_rgb_len = 1000
    vgg_len = 1000
    question_len = 1000
    for (i3d_flow, i3d_flow_seq_helper, 
                    i3d_rgb, i3d_rgb_seq_helper, 
                    vggish, vggish_seq_helper, 
                    caption, caption_seq_helper, 
                    batch_question, batch_question_seq_helper, 
                    batch_answer, batch_answer_seq_helper) in data_loader:
        #print(i3d_flow.size(1), i3d_rgb.size(1))
        i3d_flow_len = min(i3d_flow_len, i3d_flow.size(1))
        i3d_rgb_len = min(i3d_rgb_len, i3d_rgb.size(1))
        vgg_len = min(vgg_len, vggish.size(1))
        #vggish_len = vggish.size(1)
        #caption = caption.size(1)
        for question in batch_question:
            #print(question.size(1))
            question_len = min(question_len, question.size(1))
        #batch_answer = batch_answer.size(1)

    print(i3d_flow_len)
    print(i3d_rgb_len)
    print(vgg_len)
    print(question_len)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_base_path", type=Path, default="./AVSD_Jim/data")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--shuffle", type=utils.str2bool, default="True")
    parser.add_argument("--cuda", type=utils.str2bool, default="True")
    parser.add_argument("--outputDir", type=Path, default="./AVSD_Jim/output/")
    parser.add_argument("--modelType", type=str, default="SimpleModel")
    parser.add_argument("--context", type=int, default=1)
    args = parser.parse_args()
    
    train, valid, test, word = init(args)
    print('Train')
    check_data(train)
    #check_dataset(train, word)
    print('Valid')
    check_data(valid)
    #check_dataset(valid, word)
    print('Test')
    check_data(test)
    #check_dataset(test, word)
    # 
    # Save model
    #model.cpu()
    #model.save() 

if __name__ == "__main__":
    main()