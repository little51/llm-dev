from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# 从OAI_CONFIG_LIST中装载LLM配置
# 可配置多个
# 也可在程序中设定
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
# 按配置创建AI助理
assistant = AssistantAgent("assistant", llm_config={
                           "config_list": config_list})
# 创建AI Agent用于生成和运行代码
# 生成的代码存放在coding目录
# use_docker 如果设置成True，则代码将部署和运行于Docker
user_proxy = UserProxyAgent("user_proxy", code_execution_config={
                            "work_dir": "coding", "use_docker": False})
# Agent开始使用Chat方式解决用户的问题
# 只指定初始对话内容，后续的对话是Agent自我控制的
user_proxy.initiate_chat(
    assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")
