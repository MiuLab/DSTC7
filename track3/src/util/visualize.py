import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predict_path", type=str)
    args = parser.parse_args()
    with open(args.predict_path) as f:
        data = json.load(f)
    
    bar = "[*]" + "=" * 80
    for i, dialog_information in enumerate(data["dialogs"]):
        print(bar)
        print("Dialog {}".format(i))
        for j, qa_pair in enumerate(dialog_information["dialog"]):
            if j != 0:
                print(bar)
                print("Turn {}".format(j))
                print("Question:\t{}".format(qa_pair["question"]))
                print("Answer:\t{}".format(qa_pair["answer"]))
                print("Predict:\t{}".format(qa_pair["predict"]))
                print(bar)
            else:
                pass
        print(bar)

if __name__ == "__main__":
    main()