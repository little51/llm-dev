import json
import argparse
import os
from tqdm import tqdm
from tokenizations import tokenization_bert_word_level as \
    tokenization_bert


def make_tokenized(raw_data_path, tokenized_data_path, vocab_filename,
                   num_pieces, min_length):
    # 初始化tokenizer
    full_tokenizer = tokenization_bert.BertTokenizer(
        vocab_file=vocab_filename)
    full_tokenizer.max_len = 999999
    # 读取了原始数据文件 `raw_data_path` 中的内容
    # 并用 `[SEP]` 替换换行符 `\n`，以表示换行和段落结束
    with open(raw_data_path, 'r', encoding='utf8') as f:
        lines = json.load(f)
        lines = [line.replace('\n', ' [SEP] ')
                 for line in lines]
    all_len = len(lines)
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    # 根据参数 `num_pieces` 将数据分成若干段，每段处理部分数据
    for i in tqdm(range(num_pieces)):
        sublines = lines[all_len // num_pieces *
                         i: all_len // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            # 把尾部例子添加到最后一个piece
            sublines.extend(lines[all_len // num_pieces *
                                  (i + 1):])
        sublines = [full_tokenizer.tokenize(line) for line
                    in sublines if
                    len(line) > min_length]
        sublines = [full_tokenizer.convert_tokens_to_ids(
            line) for line in sublines]
        full_line = []
        for subline in sublines:
            # 在每个子句之前添加 `[MASK]` 表示子句开始
            full_line.append(full_tokenizer.convert_tokens_to_ids(
                '[MASK]'))
            full_line.extend(subline)
            # 在每个子句之后添加 `[CLS]` 表示子句结束
            full_line.append(full_tokenizer.convert_tokens_to_ids(
                '[CLS]'))
        with open(tokenized_data_path +
                  '/tokenized_train_{}.txt'.format(i), 'w') as f:
            for id in full_line:
                f.write(str(id) + ' ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path',
                        default='./data/train.json',
                        type=str, required=False)
    parser.add_argument('--tokenized_data_path',
                        default='./tokenized',
                        type=str, required=False)
    parser.add_argument('--vocab_filename',
                        default='./vocab/vocab_user.txt',
                        type=str, required=False)
    parser.add_argument('--num_pieces', default=100, type=int,
                        required=False)
    parser.add_argument('--min_length', default=128, type=int,
                        required=False)
    args = parser.parse_args()
    make_tokenized(args.raw_data_path, args.tokenized_data_path,
                   args.vocab_filename, args.num_pieces,
                   args.min_length)
