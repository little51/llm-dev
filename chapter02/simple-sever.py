from transformers import AutoTokenizer, AutoModel

# 装载tokenizer
tokenizer = AutoTokenizer.from_pretrained("Tokenizer路径")
# 装载model
model = AutoModel.from_pretrained("模型路径", device='cuda')
# 将模型设置成评估模式
model = model.eval()
# 推理
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
