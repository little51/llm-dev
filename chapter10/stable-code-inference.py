import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path = "./dataroot/models/stabilityai/stable-code-3b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
)
model.cuda()
inputs = tokenizer("import torch\nimport torch.nn as nn",
                   return_tensors="pt").to(model.device)
tokens = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.2,
    do_sample=True,
)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
