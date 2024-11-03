# 第13章 翻译模型应用

## 一、安装

```bash
conda create -n pdftrans python=3.10 -y
conda activate pdftrans 
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
```
如果是windows，且有GPU，因为默认安装的torch是CPU版的，需要卸载后从https://pytorch.org/网站上找到命令安装：

```bash
pip uninstall torch -y 
pip uninstall torchvision -y
pip uninstall torchaudio -y
# cuda 12.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# cuda 11.7
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```

## 二、下载英译中模型

```shell
python model_download.py --e --repo_id Helsinki-NLP/opus-mt-en-zh --token YPY8KHDQ2NAHQ2SG
```

## 三、测试

```shell
conda activate pdftrans 
# 用Helsinki-NLP/opus-mt-en-zh模型翻译
python pdf-trans.py --pdf  sample.pdf
# 用LLM翻译
python pdf-trans.py --pdf sample.pdf --llm
```

