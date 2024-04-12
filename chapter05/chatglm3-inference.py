from transformers import AutoTokenizer, AutoModel

model_path = "./dataroot/models/THUDM/chatglm3-6b"
# 装载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True)
# 装载model
model = AutoModel.from_pretrained(
    model_path, trust_remote_code=True)
# 将模型装入GPU
model = model.to("cuda")
# 将模型设置成评估模式
model = model.eval()
# 推理
while True:
    prompt = input("请输入问题，回车退出：")
    if prompt == "":
        break
    response, history = model.chat(tokenizer, prompt, history=[])
    print(response)
