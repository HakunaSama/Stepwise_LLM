"""
SFT数据集，是对话形式，需要注意的是，这里的lossmask只让训练数据中的模型回答内容计算loss，让模型只学习如何回答不学习如何提问。
数据格式：
 {
    "conversations":
    [
        {"role": "user", "content": "请告诉我在中国古代的“四大发明”是什么？"},
        {"role": "assistant", "content": "中国古代的“四大发明”是指造纸术、印刷术、火药和指南针。这四项发明对世界文明的发展产生了深远的影响："}
    ]
}
"""
import json
from torch.utils.data import Dataset, DataLoader
import torch
import os

#关闭hugging face tokenizers库的多线程并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SFTDataset(Dataset):
    def __init__(self,data_path,tokenizer,max_length=1024):
        super().__init__()
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.samples=self.load_data(data_path)
        self.bos_id=tokenizer('<|im_start|>assistant',add_special_tokens=False).input_ids   # 开始符的id
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids           # 结束符的id

    def __len__(self):
        return len(self.samples)

    def load_data(self,path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                try:#line：
                    # {
                    # "conversations":
                    #   [
                    #       {"role": "user", "content": "请告诉我在中国古代的“四大发明”是什么？"},
                    #       {"role": "assistant", "content": "中国古代的“四大发明”是指造纸术、印刷术、火药和指南针。这四项发明对世界文明的发展产生了深远的影响："}
                    #   ]
                    # }
                    text = json.loads(line.strip())#text为字典类型
                    samples.append(text)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue
        return samples

    def _create_chat_prompt(self,conversations):
        '''
        构建符合ChatML格式的对话，conversations为字典列表
        [
            {"role": "user", "content": "请告诉我在中国古代的“四大发明”是什么？"},
            {"role": "assistant", "content": "中国古代的“四大发明”是指造纸术、印刷术、火药和指南针。这四项发明对世界文明的发展产生了深远的影响：\n"}
        ]
        '''
        messages=[]
        for i,turn in enumerate(conversations):
            # turn: {"role": "user", "content": "请告诉我在中国古代的“四大发明”是什么？"}
            # user和assistant一对一轮，偶数是user，奇数是assistant
            role='user' if i%2==0 else 'assistant'
            messages.append({"role":role,"content":turn['content']})
            #对于我们自己的数据集，处理前后并没有区别，主要是为了应对role缺失或者和我们想要的标签不同的情况

        #返回的是str，因为tokenize=False
        #return: <|im_start|>user请告诉我在中国古代的“四大发明”是什么？<|im_end|><|im_start|>assistant中国古代的“四大发明”是指造纸术、印刷术、火药和指南针。这四项发明对世界文明的发展产生了深远的影响：<|im_end|>
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    '''
    生成和input_ids等长的loss_mask：位于<bos_id> … <eos_id>包围的区间置为1，其他token置为0
    也就是说，loss_mask把训练数据中除了模型回答的内容外全部置为0了，只有模型回答的内容被计算loss
    '''
    def _generate_loss_mask(self,input_ids):
        #先初始化一个和input_ids长度相同的loss mask
        loss_mask=[0]*len(input_ids)
        i=0
        #逐个token遍历
        while i<len(input_ids):
            #判断当前位置是否是起始标记，因为起始标记'<|im_start|>assistant'是多个token，所以需要切片比较
            if input_ids[i:i+len(self.bos_id)]==self.bos_id:
                start=i+len(self.bos_id)#起始位置
                end=start
                #从起始位置继续往后遍历，直到找到结束标记，end就是结束位置了
                while end<len(input_ids):
                    if input_ids[end:end+len(self.eos_id)]==self.eos_id:
                        break
                    end+=1
                #将loss mask的start到min(end和max_length)位置设置为1（可能没有end，那就取max_length）
                for j in range(start+1,min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j]=1
                # 如果end没有到
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        # sample为字典，内容如下
        # {
        #   "conversations":
        #   [
        #       {"role": "user", "content": "请告诉我在中国古代的“四大发明”是什么？"},
        #       {"role": "assistant", "content": "中国古代的“四大发明”是指造纸术、印刷术、火药和指南针。这四项发明对世界文明的发展产生了深远的影响："}
        #   ]
        # }
        # 获取一个样本
        sample = self.samples[index]
        # 构建该样本对话类型的str
        prompt = self._create_chat_prompt(sample['conversations'])
        # 获取tokenizer之后的输入
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 对不够长度的进行padding
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X, Y, loss_mask
