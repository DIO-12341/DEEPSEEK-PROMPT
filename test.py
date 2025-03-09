
from Deepseek_LM import DeepSeekClient
client = DeepSeekClient(
    model='deepseek-chat',  # 或 deepseek-reasoner
    api_key='sk-ba96310bb03c47a9ad0ad30c936928af'
)

# 多轮对话示例
client.prompt("什么是量子纠缠？")
response = client.prompt("请用通俗易懂的方式解释")
print("响应内容:", response)




# 查看对话历史
print("\n对话历史记录：")
print(client.get_conversation_history())
