import json
import glob

if __name__ == "__main__":
    names = glob.glob("./data/*.txt")
    train_data = []
    x = 0
    for j, name in enumerate(names):
        f = open(name, 'r+', encoding="utf-8")
        lines = [line for line in f.readlines()]
        for i in range(lines.__len__()):
            text = lines[i].strip()
            train_data.append(text)
        f.close()
        x = x + 1
        print(str(x) + " of " + str(len(names)))

    with open('./data/train.json', 'w+', encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False)
