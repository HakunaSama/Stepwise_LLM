import streamlit as st
from openai import OpenAI

# 基础配置：修改为你的服务地址、Key 和模型 ID
API_URL = "http://127.0.0.1:8000/v1"
API_KEY = "your_api_key"
MODEL_ID = "minimind"

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=API_URL)

# 页面配置和基本样式
st.set_page_config(page_title="MiniMind 聊天 Demo", layout="wide")
# 自定义简单样式
st.markdown("""
<style>
.chat-container { max-width: 800px; margin: auto; padding: 20px; background: #f0f2f6; border-radius: 8px; }
.user-bubble { background-color: #4a90e2; color: white; padding: 10px 15px; border-radius: 15px; margin: 5px 0; max-width: 70%; align-self: flex-end; }
.assistant-bubble { background-color: #ececec; color: black; padding: 10px 15px; border-radius: 15px; margin: 5px 0; max-width: 70%; align-self: flex-start; }
.chat-box { display: flex; flex-direction: column; }
.stTextInput>div>div>input { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

st.title("💬 MicroCortex")

# 初始化会话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 外层容器
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    # 渲染消息
    for msg in st.session_state.messages:
        role = msg['role']
        content = msg['content'].replace('<', '&lt;').replace('>', '&gt;')
        if role == 'user':
            st.markdown(f'<div class="chat-box" style="align-items: flex-end;"><div class="user-bubble">{content}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-box" style="align-items: flex-start;"><div class="assistant-bubble">{content}</div></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# 输入框和发送按钮在同一行
col1, col2 = st.columns([8, 1])
with col1:
    user_input = st.text_input("", placeholder="输入消息并回车发送", key="input")
with col2:
    send = st.button("发送")

# 处理输入
if send and user_input:
    # 保存用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    # 调用模型
    try:
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=st.session_state.messages
        )
        assistant_reply = resp.choices[0].message.content
    except Exception as e:
        assistant_reply = f"调用出错: {e}"
    # 保存助手回复
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    # 刷新页面
    st.experimental_rerun()