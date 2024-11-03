# 第15章 语音模型应用

## 一、环境

### 1、代码准备

```bash
git clone https://github.com/QwenLM/Qwen-Audio
cd Qwen-Audio
git checkout 2979d08
```

### 2、环境创建

```bash
conda create -n qwen-audio python=3.10 -y
conda activate qwen-audio 
# 安装基础依赖库
pip install -r requirements.txt \
-i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
# 安装web demo依赖库
pip install -r requirements_web_demo.txt \
-i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
# 校验PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

### 3、下载模型

```bash
wget https://aliendao.cn/model_download.py
conda activate qwen-audio
python model_download.py --e --repo_id Qwen/Qwen-Audio-Chat \
--token YPY8KHDQ2NAHQ2SG
```

### 4、运行Demo

```bash
python web_demo_audio.py -c dataroot/models/Qwen/Qwen-Audio-Chat \
--server-name 0.0.0.0 --cpu-only
```

## 二、后台服务

```bash
pip install -r requirements_api.txt \
-i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
python qwen_audio_api.py -c dataroot/models/Qwen/Qwen-Audio-Chat --cpu-only
```

## 三、前台应用

```bash
create-react-app audio-chat
cd audio-chat
npm i --save  @chatui/core
npm i --save openai
npm i --save localStorage
# https://github.com/2fps/recorder
npm i --save js-audio-recorder
npm start
```

