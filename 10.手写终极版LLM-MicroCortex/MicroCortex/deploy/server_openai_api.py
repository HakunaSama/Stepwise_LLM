"""
模型服务器，加载模型，并且封装FastAPI接口
模型加载：
    支持两种形式：
        直接从.pth权重pth权重+MiniMindConfig构造模型
        从预先转换好的Transformers目录加载（--load 1），可选注入LoRA低秩适配器
推理接口：
    用FastAPI实现OpenAI兼容端点 /v1/chat/completions
启动方式：
    python serve_openai_api.py --hidden_size 768 …脚本自动启动Uvicorn监听0.0.0.0：8998，本地局域网可以通过OpenAISDK/curl调用
"""
# 导入相关包
import argparse
import json
import os
import sys

__package__ = "deploy"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
import warnings
import uvicorn

from threading import Thread
from queue import Queue
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# 本地自研模型、LoRA 注入工具
from model.model_minimind import MicroCortexConfig, MicroCortexForCausalLM
from model.model_lora import apply_lora, load_lora

warnings.filterwarnings('ignore')

"""
创建FastAPI服务器
"""
app = FastAPI()

"""
加载和初始化模型
"""
def init_model(args):
    #根据启动参数加载模型，支持两种来源：
    # 原生.pth权重（args.load==0）
    # Transformers模型（args.load==1）
    if args.load == 0:
        # 从原生权重文件加载
        tokenizer = AutoTokenizer.from_pretrained('../model/')
        # 选择加载的模型类别
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason'}
        ckp = f'../{args.out_dir}/{modes[args.model_mode]}_{args.hidden_size}_moe.pth'
        # 构建模型并加载参数
        model = MicroCortexForCausalLM(MicroCortexConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            max_seq_len=args.max_seq_len,
        ))
        model.load_state_dict(torch.load(ckp, map_location=device), strict=True)
        #如果指定了LoRA，给模型先应用上LoRA，再加载LoRA参数
        if args.lora_name != 'None':
            apply_lora(model)
            load_lora(model, f'../{args.out_dir}/{args.lora_name}_{args.hidden_size}.pth')
    else:
        #直接用Transformers目录加载
        model_path = '../MicroCortex'
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    #输出可训练参数量供参考
    print(f'MicroCortex模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}(Million)')
    return model.eval().to(device), tokenizer

"""
POST /v1/chat/completions的输入的JSON Schema.可以理解为定义请求格式类
"""
class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192
    stream: bool = False
    tools: list = []

"""
自定义一个streamer：把每一段生成的token变成text后通过队列传递给主线程（前端界面、终端等等），
就是一个负责把模型生成的结果即时传递出去的中间层
"""
class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue          #一个线程安全的队列，用来存放大模型输出的文本，供给外部时使用
        self.tokenizer = tokenizer  #用来把token转成字符串

    # 每当大模型生成一段可用的字符串text，这个函数就会被自动调用
    def on_finalized_text(self, text: str, stream_end: bool = False):
        # 将text，即生成的文本，放入队列
        self.queue.put(text)
        # 如果生成结束，则放入None标记结束
        if stream_end:
            self.queue.put(None)# 用 None 标记流式结束

"""
流式SSE的响应函数，每次yield一段JSON字符串，具体是一边调用大模型生成内容，一边从queue中把生成的内容不断响应出去
"""
def generate_stream_response(messages,      # 多轮对话消息列表
                             temperature,   # 控制生成的随机性
                             top_p,
                             max_tokens):   # 限制生成的长度
    try:
        # 把对话列表变成chat_template格式，并截断到最后max_tokens个，作为prompt
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)[-max_tokens:]
        # 对prompt进行tokenizer
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        # 创建线程安全的队列
        queue = Queue()
        # 创建streamer用来将大模型生成的文本流式放入queue中
        streamer = CustomStreamer(tokenizer, queue)

        # 后台线程运行 generate，大模型根据input_ids进行生成，并且将生成结果实时调用streamer的on_finalized_text函数，将生成文本放入到queue中
        def _generate():
            with torch.no_grad():  # 禁用梯度计算，更快更省显存
                model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer
                )

        # 后台启动大模型生成线程
        Thread(target=_generate).start()

        # 循环从queue中取出生成的文本
        while True:
            text = queue.get()
            if text is None:
                '''
                yield 的作用是：每次执行到它，就“产出一个值”，然后暂停函数的运行，等下一次再从这个位置继续运行。
                什么时候从这个位置继续运行呢？就是返回的生成器对象调用next()，或者for循环的时候，就会再次从这里继续运行
                也可以隐式调用，直接对generate_stream_response(xxx)进行调用next()，或者进行for循环。
                '''
                yield json.dumps({ # json.dumps把json转成字符串
                    "choices": [{
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }, ensure_ascii=False)
                break

            # 返回生成的json，这里是openai chat api的格式，delta的意思是增量内容
            # 返回给前端，前端自己就解析了
            yield json.dumps({
                "choices": [{"delta": {"content": text}}]
            }, ensure_ascii=False)

    except Exception as e:
        yield json.dumps({"error": str(e)})

"""
FastAPI服务器实现（/v1/chat/completions）请求，当外部调用到http://127.0.0.1:8998/v1/chat/completions，
就调用了本接口
"""
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):# ChatRequest：外部请求
    try:
        # 如果请求是流式请求
        if request.stream:
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(    #遍历generate_stream_response返回的生成器对象，这里是隐式调用
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens
                )),
                media_type="text/event-stream"
            )

        # 如果请求的是一次性输出，这里我们就要自己重新处理把message喂给模型再获取模型输出的过程了
        # 也就是generate_stream_response里的过程，但是不需要多线程了，完全产出后一起返回
        else:
            # 把对话列表变成chat_template格式，并截断到最后max_tokens个，作为prompt
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True
            )[-request.max_tokens:]

            # 进行tokenizer
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

            # 大模型根据input_ids进行生成，这里没有streamer参数了，也就是说大模型不再将生成结果实时调用streamer的on_finalized_text函数，将生成文本放入到queue中
            with torch.no_grad():# 禁用梯度计算，更快更省显存
                generated_ids = model.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + request.max_tokens,
                    do_sample=True,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=request.top_p,
                    temperature=request.temperature
                )
                # 生成回答的str
                answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            #按照OpenAI风格返回JSON，这里我们要自己构建openai api格式的回复
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "microcortex",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop"
                    }
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
启动server
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server for MicroCortex")
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--lora_name', default='None', type=str)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=16, type=int)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--load', default=0, type=int, help="0: 从原生torch权重，1: 利用transformers加载")
    parser.add_argument('--model_mode', default=1, type=int,
                        help="0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型")
    #全局设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 加载模型和分词器
    model, tokenizer = init_model(parser.parse_args())
    #启动uvicorn服务（0.0.0.0：8998）
    uvicorn.run(app, host="0.0.0.0", port=8998)