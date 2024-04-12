from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 检验是否支持GPU
print(torch.__version__)
print(torch.cuda.is_available())
# 设置生成随机种子初始化网络
torch.manual_seed(1234)
# 模型路径
model_path = "./dataroot/models/Qwen/Qwen-VL-Chat-Int4/"
# 装载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          trust_remote_code=True)
# 装载模型并设置评估模式
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="cuda", trust_remote_code=True).eval()
# 输入图片与文本预处理
query = tokenizer.from_list_format([
    {'image': 'https://pic4.zhimg.com/80/v2-4e6e736dbfa729e7c61719fb3ee87047_720w.webp'},
    {'text': '这是什么'},
])
# 推理
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
