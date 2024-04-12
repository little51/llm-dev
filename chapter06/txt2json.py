import json
import argparse


def txt2json(infile, outfile):
    with open(infile, 'r', encoding='utf-8') as file:
        data = file.read()
    lines = data.split('\n')
    json_data = []
    for i in range(len(lines)):
        if (i - 2 >= 0) and ((i - 2) % 3 == 0):
            question = lines[i-2]
            answer = lines[i-1]
            json_data.append({"q": question, "a": answer})
    with open(outfile, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)
    args = parser.parse_args()
    txt2json(args.infile, args.outfile)
