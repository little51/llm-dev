# 智能代码应用

## 一、docker安装

```bash
# 更新软件源信息
sudo apt-get update
# 安装组件
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
# 添加Docker官方的GPG密钥
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
# 设置官方仓库
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
# 安装Docker
sudo apt-get update
sudo apt-get install docker-ce
# 验证安装
sudo docker --version
# 将当前用户增加到docker用户组
sudo gpasswd -a ${USER} docker
sudo service docker restart
newgrp - docker
# 验证以当前用户运行Docker镜像
docker run hello-world
```

## 二、虚拟环境安装

```bash
conda create -n autogen python=3.10
conda activate autogen
pip install pyautogen==0.2.15  \
-i https://pypi.mirrors.ustc.edu.cn/simple \
--trusted-host=pypi.mirrors.ustc.edu.cn
```

## 三、运行环境验证

```bash
conda activate autogen
python autogen-with-docker.py
```

## 四、多Agent会话

```bash
conda activate autogen
python agent-group.py
```

