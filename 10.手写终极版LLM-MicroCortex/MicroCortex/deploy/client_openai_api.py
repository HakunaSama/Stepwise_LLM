"""
一个简单的聊天客户端脚本，
演示如何通过 OpenAI 兼容接口
调用本地服务端。服务端是serve_openai_api.py。
支持：
- 流式 / 非流式两种输出
- 可配置将最近 N 轮对话携带给模型做上下文
- 自动把用户与助手回复保存在 conversation_history 中
"""


# client连接到本地服务端，也就是我们的microcortex服务器
from openai import OpenAI
#连接本地ollama（兼容openai接口）的服务端，也就是我们的minimind服务器
client = OpenAI(
    api_key="ollama",
    base_url="http://127.0.0.1:8998/v1"
)

stream = True   #是否流式输出

# 对话记录
conversation_history_origin = [] #conversation_history_origin保存完整聊天
conversation_history = conversation_history_origin.copy()   #conversation_history作为工作副本
history_messages_num = 2  # 设置为偶数（Q+A）为最近（Q+A）条消息作为上下文，为0则每次不携带历史对话进行独立QA

# 主循环
while True:
    # 读取用户输入，并且存入conversation_history
    query = input('[Question]: ')
    conversation_history.append({"role": "user", "content": query})

    # 使用client来向http://127.0.0.1:8998/v1请求/chat/completions，会被serve_openai_api.py中的
    # @app.post("/v1/chat/completions")
    # async def chat_completions(request: ChatRequest):
    # 函数处理并返回响应
    response = client.chat.completions.create(
        model="microcortex",
        messages=conversation_history[-history_messages_num:],  # 传入消息
        stream=stream   # 响应方式
    )
    if not stream:
        # 非流式则直接读取整段，也就是choices[0].message.content
        '''服务端返回格式
        {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "minimind",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop"
                }
            ]
        }
        '''
        assistant_res = response.choices[0].message.content
        print('[Answer]: ', assistant_res)
    else:
        # 流式则一边收到一边打印
        print('[Answer]: ', end='')
        assistant_res = ''
        for chunk in response:
            '''
            {
                "choices": [{"delta": {"content": text}}]
            }
            '''
            print(chunk.choices[0].delta.content or "", end="")
            assistant_res += chunk.choices[0].delta.content or ""
    # 把助手回复加入历史
    conversation_history.append({"role": "assistant", "content": assistant_res})
    print('\n\n')