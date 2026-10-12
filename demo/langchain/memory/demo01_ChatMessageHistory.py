from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import messages_to_dict, messages_from_dict
import json

# 1.创建一个ChatMessageHistory对象，用来存储对话信息
history = ChatMessageHistory()
# 添加用户消息
history.add_user_message("在吗？")
# 添加大模型消息
history.add_ai_message("在")
# 打印所有消息
print(f'history.messages-->{history.messages}')

# 2.可以将history.messages中的信息保存到字典中，然后保存到数据库或者文件中，方便后续读取
# 2.1 messages_to_dict()方法将history.messages中的信息转换成字典
dicts = messages_to_dict(history.messages)
print(f'dicts-->{dicts}')
# 2.2 这里将dicts保存到文件中
with open('history.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(dicts, indent=2, ensure_ascii=False))

# 3.从文件中读取出字典，然后将字典转换成消息
# 3.1 读取文件
messages = json.load(open('history.json', 'r', encoding='utf-8'))
# 3.2 然后将字典转换成消息
chat_messages = messages_from_dict(messages)
print(f'chat_messages-->{chat_messages}')