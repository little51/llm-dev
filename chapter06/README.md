# 大语言模型微调

## 一、ChatGLM3-6b微调

```shell
# 1、下载源码
git clone https://github.com/THUDM/ChatGLM3
cd ChatGLM3
git checkout c814a72
# 2、建立python3.10虚拟环境
conda create -n ChatGLM3 python=3.10 -y
conda activate ChatGLM3
# 3、在ChatGLM3虚拟环境下安装依赖
# 安装模型运行环境
pip install -r requirements.txt \
-i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
# 安装微调运行环境
# 修改依赖库
vi finetune_demo/requirements.txt
# 注释掉mpi4py，增加一行nltk和一行typer
#mpi4py>=3.1.5
nltk==3.8.1
typer==0.12.2
# 安装微调依赖库
pip install -r finetune_demo/requirements.txt \
-i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
# 4、验证torch环境
python -c "import torch; print(torch.cuda.is_available())"
# 5、下载模型
# 模型下载脚本从aliendao.cn首页下载
# 链接为 https://aliendao.cn/model_download.py
# linux下使用wget命令下载，windows下直接在浏览器打开链接下载
wget https://aliendao.cn/model_download.py
# 从aliendao.cn下载chatglm3-6b模型文件
python model_download.py --e --repo_id THUDM/chatglm3-6b --token YPY8KHDQ2NAHQ2SG
# 下载后的文件在./dataroot/models/THUDM/chatglm3-6b目录下
# 6、微调过程
# 微调需要16G左右的GPU内存，如果推理卡内存只有16G，则要适当降低每次装入推理卡的批量
vi finetune_demo/configs/ptuning_v2.yaml 
将 per_device_train_batch_size从原来的4改为2
# 微调
python finetune_demo/finetune_hf.py ./data  \
./dataroot/models/THUDM/chatglm3-6b \
finetune_demo/configs/ptuning_v2.yaml
# 测试
python finetune_demo/inference_hf.py output/checkpoint-3000 --prompt 高血压要注意哪些方面？
# 原模型测试
python finetune_demo/inference_hf.py \
./dataroot/models/THUDM/chatglm3-6b \
--prompt 高血压要注意哪些方面？
```

## 二、LLama-2-hf-chat微调

```shell
# 1、代码准备
git clone https://github.com/facebookresearch/llama-recipes
cd llama-recipes
git checkout 95418fc
# 2、环境创建
conda create -n llama-recipes python=3.10 -y
conda activate llama-recipes
pip install -U pip setuptools
pip install -r requirements.txt \
-i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
# 3、验证pytorch是否安装成功
python -c "import torch; print(torch.cuda.is_available())"
# 4、使用源码安装llama-recipes模块
pip install -e . -i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
# peft >= v0.10.0 微调时会报以下错误：cannot import name \
# prepare_model_for_int8_training' from 'peft'  （prepare_model_for_int8_training方法在peft V0.10.0中被删除了）
# 需要降低peft到v0.9.0
pip install peft==0.9.0 -i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
# 5、下载模型
# 模型下载脚本从aliendao.cn首页下载
# 链接为 https://aliendao.cn/model_download.py
# linux下使用wget命令下载，windows下直接在浏览器打开链接下载
wget https://aliendao.cn/model_download.py
# 从aliendao.cn下载Llama-2-7b-chat-hf 模型文件
python model_download.py --e --repo_id NousResearch/Llama-2-7b-chat-hf \
--token YPY8KHDQ2NAHQ2SG
# 下载后的文件在./dataroot/models/NousResearch/Llama-2-7b-chat-hf 目录下
# 7、微调过程
CUDA_VISIBLE_DEVICES=0 nohup torchrun \
--nnodes 1 --nproc_per_node 1 recipes/finetuning/finetuning.py \
--model_name ./dataroot/models/NousResearch/Llama-2-7b-chat-hf \
--output_dir output/PEFT/model \
--dataset alpaca_dataset \
--use_peft \
--peft_method lora \
--batch_size_training 4 \
--num_epochs 3 \
--context_length 1024 \
--quantization > train.log  2>&1 &
# 日志查看
tail -f train.log
```

## 三、Gemma-2b微调

```shell
# 1、建立一个python3.10虚拟环境
conda create -n gemma python=3.10 -y
conda activate gemma
# 2、建立requirements.txt，内容为
torch==2.0.1
transformers==4.38.1
accelerate==0.27.2
bitsandbytes==0.42.0
trl==0.7.11
peft==0.8.2
# 3、安装依赖库
pip install -r requirements.txt \
-i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
# 4、下载模型
wget https://aliendao.cn/model_download.py
python model_download.py --e --repo_id alpindale/gemma-2b \
--token YPY8KHDQ2NAHQ2SG
# 下载后的文件在./dataroot/models/alpindale/gemma-2b目录下
# 5、下载数据集
python model_download.py --repo_type dataset --repo_id \
Abirate/english_quotes
# 6、微调
python gemma-train.py
```

