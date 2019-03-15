from src.model.model import DialogModel
from src.dataset.dataset import AVSD_Dataset
from src.util import utils
from src.log import log
from src.util.WordHelper import WordHelper
import argparse
import json
from pathlib import Path
import copy
def init(args):
    # Load word helper
    word_helper = WordHelper(args.data_base_path / "word_embedding" / "glove.6B.300d.txt", "glove",
                                cacheDir=args.data_base_path / "cache",
                                timestamp=args.parameter_timestamp,
                                word_embed_size=300,
                                requires_grad=False)
    dataset = AVSD_Dataset(args.data_base_path, args.parameter_timestamp, word_helper, args.data_category, args.undisclose, video_upsample=args.video_upsample)
    model = DialogModel(dataset, 
                        args.batch_size, 
                        False, 
                        args.outputDir, 
                        args.modelType, 
                        word_helper, 
                        args.parameter_timestamp,
                        args.context)
    model.load()
    if args.cuda:
        model.cuda()

    return model, dataset

def dump_answer_along_with_ground_truth(args, answers):
    if not args.undisclose:
        with (args.data_base_path / "textdata" / (args.data_category + "_set.json")).open("r") as f:
            data = json.load(f)
    else:
        with (args.data_base_path / "textdata" / ("test_set4DSTC7-AVSD.json")).open('r') as f:
            data = json.load(f)

    assert len(answers) == len(data["dialogs"])
    # Append predicted answers
    for dialog_information, dialog_answers in zip(data["dialogs"], answers):
        for i, dialog in enumerate(dialog_information["dialog"]):
            dialog["predict"] = " ".join(dialog_answers[i])
    
    if args.undisclose:
        # Filtered out QA after undisclose
        data = parse_undisclose(data)
        # Replace UNDISCLOSE by our model's result
        replaced_data = replace_undisclose(data)

    # Dump json
    timestamp = args.parameter_timestamp
    fname = "test_set_predicted_{}.json".format(timestamp)
    with (args.outputDir / "generate" / fname).open('w') as f:
        json.dump(data, f, indent=4)
    
    if args.undisclose:
        fname = "test_set_predicted_{}_standard.json".format(timestamp)
        with (args.outputDir / "generate" / fname).open('w') as f:
            json.dump(replaced_data, f, indent=4)

def dump_metrics_dict(args, metrics_dict):
    fname = "metrics_dict_{}".format(args.parameter_timestamp)
    with (args.outputDir / "metrics" / fname).open('w') as f:
        json.dump(metrics_dict, f, indent=4)

def parse_undisclose(data):
    '''
        Input: 
            a data with predicted answer
        Output:
            a data after being filtered out the answer after "__UNDISCLOSED__"
    '''
    undisclose_str = "__UNDISCLOSED__"
    filtered_data = copy.deepcopy(data)

    for dialog_information in filtered_data["dialogs"]:
        for qa_idx, qa_pair in enumerate(dialog_information["dialog"]):
            # Loop over qa pair
            if qa_pair["answer"] == undisclose_str:
                # Slice dialog to this point
                dialog_information["dialog"] = dialog_information["dialog"][:qa_idx+1] # +1 is because include __UNDISCLOSE__ in it
                break

    return filtered_data

def replace_undisclose(filtered_data):
    '''
        Input: 
            a data with predicted answer
        Output:
            a data after being filtered out the answer after "__UNDISCLOSED__"
    '''
    undisclose_str = "__UNDISCLOSED__"
    replaced_data = copy.deepcopy(filtered_data)

    for dialog_information in replaced_data["dialogs"]:
        for qa_idx, qa_pair in enumerate(dialog_information["dialog"]):
            if qa_pair["answer"] == undisclose_str:
                # Replace undisclose as our model's predict result
                dialog_information["dialog"][qa_idx]["answer"] = qa_pair["predict"]
            # Pop predict result
            dialog_information["dialog"][qa_idx].pop("predict")
    return replaced_data    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_base_path", type=Path, default="./AVSD_Jim/data")
    parser.add_argument("--data_category", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cuda", type=utils.str2bool, default="True")
    parser.add_argument("--outputDir", type=Path, default="./AVSD_Jim/output/")
    parser.add_argument("--modelType", type=str, default="SimpleModel")
    parser.add_argument("--context", type=int, default=1)
    parser.add_argument("--undisclose", action="store_true")
    parser.add_argument('--video_upsample', action='store_true', default=False)
    parser.add_argument("parameter_timestamp", type=str)
    args = parser.parse_args()

    model, dataset = init(args)
    # Get answers
    log.logging.debug("[*] Model Predicting")
    answers, metrics_dict = model.predict(dataset, args.undisclose)
    log.logging.debug("[-] Model Done Predicting")
    # Dump answers with ground truth
    log.logging.debug("[*] Dumping Predicted Result into JSON")
    dump_answer_along_with_ground_truth(args, answers)
    log.logging.debug("[-] Done Dumping JSON")
    # Dump metrics_dict
    if not args.undisclose:
        log.logging.debug("[*] Dumping Metrics Dict into JSON")
        dump_metrics_dict(args, metrics_dict)
        log.logging.debug("[-] Done Dumping JSON")


if __name__ == "__main__":
    main()
