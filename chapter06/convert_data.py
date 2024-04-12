import json
import argparse


def load_data(infile):
    with open(infile, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def convert_data_conversations(original_data, trainfile, devfile):
    output_train = []
    dev_train = []
    train_data_len = round(len(original_data) * 0.7)
    i = 0
    for item in original_data:
        conversation = {
            "conversations": [
                {
                    "role": "user",
                    "content": item["q"]
                },
                {
                    "role": "assistant",
                    "content": item["a"]
                }
            ]
        }
        i = i + 1
        if train_data_len < i:
            output_train.append(conversation)
        else:
            dev_train.append(conversation)
    with open(trainfile, 'w', encoding='utf-8') as json_file:
        json.dump(output_train, json_file,
                  ensure_ascii=False, indent=4)
    with open(devfile, 'w', encoding='utf-8') as json_file:
        json.dump(dev_train, json_file,
                  ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--trainfile', type=str, required=True)
    parser.add_argument('--devfile', type=str, required=True)
    args = parser.parse_args()
    original_data = load_data(args.infile)
    convert_data_conversations(original_data,
                               args.trainfile, args.devfile)
