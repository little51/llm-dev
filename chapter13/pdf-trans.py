from bs4 import BeautifulSoup
from tqdm import tqdm
import torch
import os
import argparse
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = None
tokenizer = None
base_url = "http://server-llm-dev:8000/v1/"
client = OpenAI(api_key="EMPTY", base_url=base_url)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def translate_text_llm(text):
    messages = [
        {
            "role": "system",
            "content": "你是一个人工智能翻译工具，请翻译用户给出的英文，只输出翻译的最终结果，如不能翻译，请原样输出",
        },
        {
            "role": "user",
            "content": text
        }
    ]
    response = client.chat.completions.create(
        model="chatglm3-6b",
        messages=messages,
        stream=False,
        max_tokens=2048,
        temperature=1,
        presence_penalty=1.1,
        top_p=0.8)
    if response:
        content = response.choices[0].message.content
        return content
    else:
        print("Error:", response.status_code)
        return text


def load_trans_model():
    global model, tokenizer
    modelpath = "./dataroot/models/Helsinki-NLP/opus-mt-en-zh"
    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    model = AutoModelForSeq2SeqLM.from_pretrained(modelpath).to(device)


def translate_text(text):
    tokenized_text = tokenizer(
        text, return_tensors='pt', truncation=True,
        padding=True).to(device)
    tokenized_text['repetition_penalty'] = 2.85
    translation = model.generate(**tokenized_text)
    translated_text = tokenizer.batch_decode(
        translation, skip_special_tokens=True)
    return translated_text[0]


def pdf2html(pdf):
    if not os.path.exists(pdf):
        return False
    outfile = "output.html"
    if os.path.exists(outfile):
        os.remove(outfile)
    cmd = "pdf2htmlEX --zoom 1.5 " + pdf + " " + outfile
    os.system(cmd)
    return os.path.exists(outfile)


def translate_html(pdf, html, llm):
    with open('output.html', 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    divs = soup.find_all('div', class_="t")
    pbar = tqdm(total=len(divs))
    for div in divs:
        pbar.update(1)
        # 用div的class属性判断跳过哪些div
        skip_flag = False
        for x in range(20, 51):
            if "m" + str(x) in div["class"]:
                skip_flag = True
                break
        if skip_flag:
            continue
        # 取div内完整文字，忽略span
        text = div.get_text(strip=True, separator=' ')
        # 翻译div内容，回写到div.string
        if text is not None and text != "" and len(text) > 5:
            if llm:
                _text = translate_text_llm(text)
            else:
                _text = translate_text(text)
            div.string = _text
    with open(html, 'w', encoding='utf-8') as f:
        f.write(str(soup))


def translate_pdf_html(pdf, html, llm):
    if model is None:
        load_trans_model()
    if pdf2html(pdf):
        translate_html(pdf, html, llm)
        return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', default=None, type=str, required=True)
    parser.add_argument('--html', default="translated.html",
                        type=str, required=False)
    parser.add_argument('--llm', action='store_true',
                        default=False, required=False)
    args = parser.parse_args()
    translate_pdf_html(args.pdf, args.html, args.llm)
