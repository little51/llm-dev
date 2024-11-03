# 第18章 AI小镇应用

基于国产模型的AI小镇，使用react、phaser3、grid-engine、chatglm6b等开发。

在线demo：[https://gitclone.com/ai-town/](https://gitclone.com/ai-town/)

## 一、安装

```bash
1、安装最新版的node.js
2、git clone https://github.com/little51/llm-dev
3、cd chapter18/ai-town
4、npm i
```

## 二、运行

npm start

访问：http://localhost:3000/ai-town/

## 三、编译

npm run build

编译后生成build文件夹，将文件复制到nginx的html目录下。

## 四、功能

1、点击鼠标移动角色

2、角色与NPC相遇开始聊天，聊天内容由chatglm6b生成

3、将实现完整的角色在小镇的日常生活，工作生活计划均由大模型产生

## 五、感谢

游戏是在 https://github.com/blopa/top-down-react-phaser-game 的基础上进行了修改的，由于侧重于GPT功能，所以删除用不到的功能，增加了鼠标点击、对话等功能。