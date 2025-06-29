import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import ast

#关闭hugging face tokenizers库的多线程并行处理
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

#预训练数据集定义
class PretrainDataset(Dataset):
    #关于max_length，现在主流大模型太大了，我们无法复现，只能给出指导意见
    #小模型(<1B)：通常设置512-2048
    #中等模型(1B-7B)：通常设置2048-4096
    #大模型(13B+)：训练时设定为4096-8192，微调和推理时可能支持更大(8k-32k)
    def __init__(self, data_path, tokenizer,max_length=512):
        super().__init__()
        self.data_path=data_path
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.samples=self.load_data(self.data_path)

    def load_data(self,path):
        samples=[]
        with open(path,'r',encoding='utf-8') as f:
            for line_idx,line in enumerate(f):
                try:
                    text=json.loads(line.strip())
                    samples.append(text)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample=self.samples[index]

        #对文本进行分词
        encoding=self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',       #如果文本太短，就补 pad 到 max_length 长度
            truncation=True,            #如果文本太长，就截断到 max_length
            return_tensors='pt'          #输出格式为 PyTorch tensor（pt 是 pytorch 的简称）
        )
        #最终的 encoding 是一个字典，结构如下
        #{
            #'input_ids': tensor([...]),         # 编码后的 token id
            #'attention_mask': tensor([...]),    # mask 掩码（1 表示有效，0 表示 padding）
            # 有些模型还有 token_type_ids 等字段
        #}

        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask

if __name__=="__main__":
    pass