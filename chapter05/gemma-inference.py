import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def generate(load_in_8bit, load_in_4bit):
    model_path = "./dataroot/models/alpindale/gemma-2b"
    quantization_config = None
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if quantization_config is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=quantization_config)
    while True:
        prompt = input("请输入您的问题：")
        if prompt == "":
            break
        input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**input_ids)
        print(tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    print("torch cuda:", torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_in_8bit', default=False,
                        action='store_true', required=False)
    parser.add_argument('--load_in_4bit', default=False,
                        action='store_true', required=False)
    args = parser.parse_args()
    generate(args.load_in_8bit, args.load_in_4bit)