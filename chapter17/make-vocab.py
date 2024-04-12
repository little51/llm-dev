import argparse
import thulac
import json
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer


def make_vocab(raw_data_path, vocab_file, vocab_size):
    lac = thulac.thulac(seg_only=True)
    tokenizer = Tokenizer(num_words=vocab_size)
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        lines = json.load(f)
        for i, line in enumerate(tqdm(lines)):
            try:
                lines[i] = lac.cut(line, text=True)
            except:
                lines[i] = ''
    tokenizer.fit_on_texts(lines)
    vocab = list(tokenizer.index_word.values())
    pre = ['[SEP]', '[CLS]', '[MASK]', '[PAD]', '[UNK]']
    vocab = pre + vocab
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for word in vocab[:vocab_size + 5]:
            f.write(word + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path',
                        default='./data/train.json',
                        type=str, required=False)
    parser.add_argument('--vocab_file',
                        default='./vocab/vocab_user.txt',
                        type=str, required=False)
    parser.add_argument('--vocab_size', default=50000,
                        type=int, required=False)
    args = parser.parse_args()
    make_vocab(args.raw_data_path, args.vocab_file,
               args.vocab_size)
