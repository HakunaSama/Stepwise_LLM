"""
DPO数据集
"""
import json
from torch.utils.data import Dataset, DataLoader
import torch
import os

#关闭hugging face tokenizers库的多线程并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                '''
                数据格式为：
                {
	                "chosen": 
		            [
			            {"content": "How many moles of HBr are required to react with 2 moles of C2H6 to form 2 moles of C2H5Br along with 2 moles of H2?", "role": "user"}, 
			            {"content": "To determine the number of moles of HBr required to react with 2 moles of C2H6 (ethane) to form 2 moles of C2H5Br.", "role": "assistant"}
		            ], 
	                "rejected": 
		            [
			            {"content": "How many moles of HBr are required to react with 2 moles of C2H6 to form 2 moles of C2H5Br along with 2 moles of H2?", "role": "user"}, 
			            {"content": "To answer this question, we need to write down the chemical equation representing the reaction between hydrogen.", "role": "assistant"}
		            ]
                }
                '''
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        # chosen:[
		# 	{"content": "How many moles of HBr are required to react with 2 moles of C2H6 to form 2 moles of C2H5Br along with 2 moles of H2?", "role": "user"},
		# 	{"content": "To determine the number of moles of HBr required to react with 2 moles of C2H6 (ethane) to form 2 moles of C2H5Br.", "role": "assistant"}
		# ],
        chosen = item['chosen']  # 是一个 list，里面包含若干 {role, content}

        # rejected:[
        #     {"content": "How many moles of HBr are required to react with 2 moles of C2H6 to form 2 moles of C2H5Br along with 2 moles of H2?","role": "user"},
        #     {"content": "To answer this question, we need to write down the chemical equation representing the reaction between hydrogen.","role": "assistant"}
        # ]
        rejected = item['rejected']

        # 构建对话类型的str
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        #编码
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)#生成loss mask
        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)#生成loss mask

        #正样本的输入、标签、loss掩码
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        #负样本的输入、标签、loss掩码
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    '''
        生成和input_ids等长的loss_mask：位于<bos_id> … <eos_id>包围的区间置为1，其他token置为0
        也就是说，loss_mask把训练数据中除了模型回答的内容外全部置为0了，只有模型回答的内容被计算loss
    '''
    def _generate_loss_mask(self, input_ids):
        # 先初始化一个和input_ids长度相同的loss mask
        loss_mask = [0] * len(input_ids)
        i = 0
        # 逐个token遍历
        while i < len(input_ids):
            # 判断当前位置是否是起始标记，因为起始标记'<|im_start|>assistant'是多个token，所以需要切片比较
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # 从起始位置继续往后遍历，直到找到结束标记，end就是结束位置了
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 将loss mask的start到min(end和max_length)位置设置为1（可能没有end，那就取max_length）
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                # 如果end没有到
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask