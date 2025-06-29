- 此项目收到MiniMind项目启发，将之前所学习内容全部应用起来，从0开始构建轻量大模型。
- MicroCortex模型架构使用完全从零开始构建构建的旋转位置编码、RMS归一化、共享混合专家(MoE)、多头潜在注意力(MLA)、混合精度训练，这些技术原理大部分在前面都有对应章节的详细讲解，如果你直接从当前开始看起，并且对某一部分内容具有疑惑，请参考前面章节的详细讲解。当前项目只进行预训练，监督微调(SFT)、LoRA微调、直接偏好学习(DPO)、模型蒸馏算法请在学习完本项目后往后继续学习。
- MicroCortex项目所有核心代码均从0开始使用pytorch原生构建，不依赖第三方库提供的抽象接口
- MicroCortex同时扩展了视觉多模态模型：MicroCortex-V。
  该项目中大部分都有同名的.py、.ipynb文件对，其中.ipynb具备更加详细的注释，学习时请先阅读.ipynb文件，运行时请使用.py文件
  Pretrain数据
  本项目预训练数据集采用的是MiniMind项目所提供的开源预训练数据集：[数据集下载链接](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)（创建./dataset目录）并放到./dataset下

学习时的项目文件阅读顺序：

model_microcortex.ipynb文件中是MicroCortex模型及其配置类的定义，使用了MoE、MLA架构

llm_pretrain_dataset.py文件中是预训练数据集文件，这个比较简单

train_pretrain.ipynb文件中是MicroCortex模型的训练流程，其中包括wandb的使用，ddp分布式训练的使用
运行项目时的顺序：

首先下载数据集到./dataset文件夹，

运行命令安装相关依赖；

```shell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

执行train_pretrain.py文件进行预训练，得到训练文件pretrain_512.pth

```
python train_pretrain.py
```

用训练得到的pretrain_512.pth权重进行指令微调，得到微调模型full_sft_512.pth

```
python train_full_sft.py
```

测试模型效果

```
python eval_model.py --model_mode 1 # 默认为0：测试pretrain模型效果，设置为1：测试full_sft模型效果
```

强化学习

```
python train_dpo.py
```

