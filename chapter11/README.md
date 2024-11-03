# 第11章 VSCode插件开发

## 一、创建项目

```shell
# node.js安装
# 安装 Yeoman
npm install -g yo
yo code
? What type of extension do you want to create? New Extension (TypeScript)【回车确认】
? What's the name of your extension? stable-coder【项目名称】
? What's the identifier of your extension? stable-coder【项目标识】
? What's the description of your extension?【项目描述】
? Initialize a git repository? No【初始化git仓库】
? Bundle the source code with webpack? Yes【用webpack打包】
? Which package manager to use? npm【包管理工具】
```

## 二、修改项目配置

vscode engines版本号

"engines": {
  "vscode": "^1.70.3"   #vscode 版本高于或等于插件要求的最低版本
 }

## 三、调试

```shell
# F5调试
# Ctrl + Shift + P
# 输入helloWorld搜索
# 在调试控制台显示Congratulations, your extension "stable-coder" is now active!则说明成功
```

## 四、extension.ts说明

```shell
1、activate和deactivate
2、console.log输出日志到调试控制台
3、使用registerCommand注册启动命令
4、当执行命令时，调用showInformationMessage在VS Code的右下角显示信息
```

## 五、修改激活事件

activationEvents去掉原来那行，增加以下行：

"activationEvents": [
    "onLanguage:python",
    "onLanguage:typescript",
    "onLanguage:javascript",
    "onLanguage:java",
    "onLanguage:c",
    "onLanguage:cpp",
    "onLanguage:go"
  ],

## 六、加入openai调用

```shell
npm install openai -s
```

## 七、发布

https://azure.microsoft.com/zh-cn/products/devops/