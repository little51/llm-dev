# 第12章 检索增强应用

## 一、虚拟环境安装

```shell
conda create -n rag python=3.10 -y
conda activate rag 
pip install -r requirements.txt \
-i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
```

## 二、下载向量化模型文件

```shell
# 模型下载脚本从aliendao.cn首页下载
# 链接为 https://aliendao.cn/model_download.py
# linux下使用wget命令下载，windows下直接在浏览器打开链接下载
wget https://aliendao.cn/model_download.py
# 从aliendao.cn下载text2vec-base-chinese模型文件
python model_download.py --e --repo_id shibing624/text2vec-base-chinese \
--token YPY8KHDQ2NAHQ2SG
# 下载后的文件在./dataroot/models/shibing624/text2vec-base-chinese目录
```

## 三、运行

```shell
python rag-demo.py
```

