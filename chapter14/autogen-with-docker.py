from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# LLM配置
config_list = [
    {
        "model": "ChatGLM3-6B",
        "base_url": "http://server-llm-dev:8000/v1",
        "api_type": "open_ai",
        "api_key": "NULL",
    }
]

# assistant（AI Agent）生成
assistant = AssistantAgent("assistant", llm_config={
                           "config_list": config_list})
# UserProxyAgent代表人类发布任务
user_proxy = UserProxyAgent(
    name="agent",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "_output", "use_docker": "python:3"},
    llm_config={"config_list": config_list},
    system_message=""""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet."""
)
# 开启初始对话开始处理任务
user_proxy.initiate_chat(
    assistant, message="写一个python版本的hello world程序并运行")
