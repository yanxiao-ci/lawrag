"""
    1.ollama回顾
"""

# 2.定义函数
def dm01():
    import ollama
    response = ollama.chat(
        model="qwen3.5:4b",
        messages=[
            {"role": "user", "content": "为什么天空是蓝色的？"}
        ]
    )
    print(response)
    print("=" * 50)
    print(response.message.content)


def dm02():
    # 可以远程调用ollama
    from ollama import Client
    # client = Client(host='http://192.168.1.100:11434')
    client = Client(host='http://127.0.0.1:11434')

    response = client.chat(model='qwen3.5:4b', messages=[
        {
            'role': 'user',
            'content': '为什么海水是蓝色的？',
        }
    ])
    print(response['message']['content'])

def dm03():
    import ollama

    stream = ollama.chat(
        model='qwen3.5:4b',
        messages=[{'role': 'user', 'content': '为什么天空是蓝色的？'}],
        stream=True  # 指定流式
    )
    # 流式输出的意义：提升用户体检
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

def dm04():
    import requests

    url = f"http://127.0.0.1:11434/api/chat"
    model = "qwen3.5:4b"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,  # 模型选择
        "options": {
            "temperature": 0.  # 为0表示不让模型自由发挥，输出结果相对较固定，>0的话，输出的结果会比较放飞自我
        },
        "stream": False,  # 流式输出
        "messages": [{
            "role": "user",
            "content": "你是谁？"
        }]  # 对话列表
    }
    response = requests.post(url, json=data, headers=headers, timeout=60)
    res = response.json()
    print(res)
    print(type(res))
    print("=" * 50)
    # print(res.message.content)
    print(res['message']['content'])

# 3.测试
if __name__ == '__main__':
    # dm01()
    dm02()
    # dm03()
    # dm04()
