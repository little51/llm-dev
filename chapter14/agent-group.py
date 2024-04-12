import autogen


def TestGroupChat():
    config_list_gpt = [
        {
            "model": "ChatGLM3-6B",
            "base_url": "http://server-llm-dev:8000/v1",
            "api_type": "open_ai",
            "api_key": "NULL",
        }
    ]
    llm_config = {"config_list": config_list_gpt,
                  "seed": 42, "stream": True, "timeout": 120}
    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        system_message="A human admin.",
        code_execution_config={"last_n_messages": 2, 
        "work_dir": "groupchat"},
        human_input_mode="TERMINATE"
    )
    coder = autogen.AssistantAgent(
        name="Coder",
        llm_config=llm_config,
    )
    pm = autogen.AssistantAgent(
        name="Product_manager",
        system_message="Creative in software product ideas.",
        llm_config=llm_config,
    )
    groupchat = autogen.GroupChat(
        agents=[user_proxy, coder, pm], messages=[], max_round=12)
    manager = autogen.GroupChatManager(
        groupchat=groupchat, llm_config=llm_config)
    user_proxy.initiate_chat(
        manager, message="在百度(https://baidu.com)上查找有关gpt-4的最新论文，并找到其在软件中的潜在应用")


if __name__ == '__main__':
    TestGroupChat()
