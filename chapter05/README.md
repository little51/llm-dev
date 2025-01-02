# 第5章 大语言模型安装

## 一、ChatGLM3-6B安装

```shell
# 下载源码
git clone https://github.com/THUDM/ChatGLM3
cd ChatGLM3
git checkout c814a72
# 建立python3.10虚拟环境并激活
conda create -n ChatGLM3 python=3.10 -y
conda activate ChatGLM3
# 安装依赖库
pip install -r requirements.txt  \
-i https://pypi.mirrors.ustc.edu.cn/simple
# 降级Transformers
pip install transformers==4.38.1 \
-i https://pypi.mirrors.ustc.edu.cn/simple
# 降级sentence_transformers
pip install sentence_transformers==2.4.0  \
-i https://pypi.mirrors.ustc.edu.cn/simple
# 校验PyTorch
python -c "import torch; print(torch.cuda.is_available())"
# 下载模型
wget https://aliendao.cn/model_download.py
python model_download.py --e --repo_id THUDM/chatglm3-6b \
--token YPY8KHDQ2NAHQ2SG
# 运行推理
python chatglm3-inference.py
```

## 二、 Qwen-VL-Chat-Int4安装

```shell
# 建立python3.10虚拟环境并激活
conda create -n Qwen-VL-Chat python=3.10 -y
conda activate Qwen-VL-Chat
# 建立测试目录
mkdir Qwen-VL-Chat-Int4
cd Qwen-VL-Chat-Int4
# 获取下载脚本
wget https://aliendao.cn/model_download.py
# 安装脚本依赖库
pip install transformers==4.32.0 \
-i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
# 下载模型
python model_download.py --e --repo_id Qwen/Qwen-VL-Chat-Int4 --token YPY8KHDQ2NAHQ2SG
# 安装依赖库
pip install -r ./dataroot/models/Qwen/Qwen-VL-Chat-Int4/requirements.txt  \
-i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
# 安装optimum和gekko库
pip install optimum gekko \
-i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
# 降级Transformers
pip install transformers==4.38.1 \
-i https://pypi.mirrors.ustc.edu.cn/simple
# 降级sentence_transformers
pip install sentence_transformers==2.4.0  \
-i https://pypi.mirrors.ustc.edu.cn/simple
# 校验PyTorch
python -c "import torch; print(torch.cuda.is_available())"
# PyTorch in CUDA 11.7
pip uninstall torch -y 
pip uninstall torchvision -y
# 比如在linux下，针对cuda11.7，用以下命令
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
-i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
# 安装AutoGPTQ库
git clone https://github.com/JustinLin610/AutoGPTQ.git
cd AutoGPTQ
pip install -v . \
-i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
# 安装完成后回退到Qwen-VL-Chat-Int4目录
cd ..
# 运行
python qwen-vl-inference.py
# 12G GPU内存
```

## 三、Llama-2-7b-chat安装

```shell
# 获取text-generation-webui源码，并将源码固定到指定的版本
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
git checkout 1934cb6
# 创建python3.10虚拟环境
conda config --add envs_dirs /nova_compute_data/anaconda3
conda create -n text-generation-webui python=3.10 -y
# 编辑requirements.txt，只保留tiktoken及以前的库
# 安装依赖库
conda activate text-generation-webui
pip install -r requirements.txt \
-i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
# 校验PyTorch
python -c "import torch; print(torch.cuda.is_available())"
# 下载模型
# 模型下载脚本从aliendao.cn首页下载
# 链接为 https://aliendao.cn/model_download.py
# linux下使用wget命令下载，windows下直接在浏览器打开链接下载
wget https://aliendao.cn/model_download.py
# 从aliendao.cn下载Llama-2-7b-chat-hf 模型文件
python model_download.py --e --repo_id NousResearch/Llama-2-7b-chat-hf \
--token YPY8KHDQ2NAHQ2SG
# 下载后的文件在./dataroot/models/NousResearch/Llama-2-7b-chat-hf 目录下
# 移动模型
mkdir ./models/Llama-2-7b-chat-hf
mv ./dataroot/models/NousResearch/Llama-2-7b-chat-hf/* ./models/Llama-2-7b-chat-hf/
# 运行
python server.py --listen-host 0.0.0.0 --listen --load-in-4bit
# 中文模型
python model_download.py --e --repo_id LinkSoul/Chinese-Llama-2-7b \
--token YPY8KHDQ2NAHQ2SG
mkdir ./models/Chinese-Llama-2-7b
mv ./dataroot/models/LinkSoul/Chinese-Llama-2-7b/* ./models/Chinese-Llama-2-7b/
python server.py --listen-host 0.0.0.0 --listen
```

## 四、Gemma-2b安装

```shell
# 创建虚拟环境
conda create -n gemma python=3.10 -y
conda activate gemma
mkdir gemma-2b
cd gemma-2b
# 安装依赖库
pip install -r requirements.txt \
-i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
# 下载模型
# 模型下载脚本从aliendao.cn首页下载
# 链接为 https://aliendao.cn/model_download.py
# linux下使用wget命令下载，windows下直接在浏览器打开链接下载
wget https://aliendao.cn/model_download.py
# 从aliendao.cn下载gemma-2b模型文件
python model_download.py --e --repo_id alpindale/gemma-2b --token YPY8KHDQ2NAHQ2SG
# 下载后的文件在./dataroot/models/alpindale/gemma-2b目录下
# 运行
# 全精度
python gemma_sample.py
# 8-bit量化
python gemma_sample.py --load_in_8bit
# 4-bit量化
python gemma_sample.py --load_in_4bit
```

## 五、Whisper-Large-V3安装

```shell
# 创建虚拟环境
conda create -n Whisper-Large-V3 python=3.10 -y
conda activate Whisper-Large-V3
mkdir Whisper-Large-V3
cd Whisper-Large-V3
# 安装依赖库
pip install -r requirements_whisper.txt \
-i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
# 校验PyTorch
python -c "import torch; print(torch.cuda.is_available())"
# 下载模型
# 模型下载脚本从aliendao.cn首页下载
# 链接为 https://aliendao.cn/model_download.py
# linux下使用wget命令下载，windows下直接在浏览器打开链接下载
wget https://aliendao.cn/model_download.py
# 从aliendao.cn下载gemma-2b模型文件
python model_download.py --e --repo_id openai/whisper-large-v3 --token YPY8KHDQ2NAHQ2SG
# 下载后的文件在./dataroot/models/openai/whisper-large-v3目录下
# 运行
python whisper-audio.py
```

