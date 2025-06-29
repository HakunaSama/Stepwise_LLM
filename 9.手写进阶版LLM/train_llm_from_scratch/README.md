## 开始训练
### 直接运行
预训练:
python train.py
SFT:
python sft_train.py

### torchrun
预训练:
torchrun --nproc_per_node=2 train.py
SFT:
torchrun --nproc_per_node=2 sft_train.py

### deepspeed
预训练:
deepspeed --include 'localhost:0,1' train.py\
SFT:
deepspeed --include 'localhost:0,1' sft_train.py

## 测试
test_llm.ipynb

这里的模型和训练过程仅供学习，这里的训练数据还并不能真的训练出一个可用的预训练模型，这里只使用了最基本的MLP层和注意力层，并且这里提前示范了SFT微调和DPO强化学习，可以先不要关注，SFT和DPO的理论和实际实现在后面有更加详细的讲解和实现。