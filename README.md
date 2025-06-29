<div align="center">
  <h3>"从零开始手写大模型"</h3>
</div>

本项目旨在记载大模型学习历程，对一些当下的大模型相关技术算法进行了详细原理讲解和代码复现。

本项目中包含两个重要项目**手写终极版LLM：MicroCortex**、**手写终极版VLM：MicroCortex-V**，从零开始实现了轻量化的LLM和VLM，并且实现了预训练、指令监督微调、LoRA微调、DPO强化学习、模型蒸馏、推理训练全流程。虽然训练数据集采用开源数据集，但所有核心算法代码均使用pytorch原生构建，不使用第三方库提供的抽象接口。

# 📌项目目录

项目目录

1. [手写SelfAttention](./1.手写SelfAttention)
2. [手写MultiHeadAttention](./2.手写MultiHeadAttention)
3. [手写GroupQueryAttention](./3.手写GroupQueryAttention)
4. [手写TransformerDecoder](./4.手写TransformerDecoder)
5. [手写基础版LLM](./5.手写基础版LLM)
6. [手写MOE](./6.手写MOE)
7. [手写旋转位置编码](./7.手写旋转位置编码)
8. [手写MLA](./8.手写MLA)
9. [手写进阶版LLM](./9.手写进阶版LLM)
10. [手写终极版LLM-MicroCortex](./手写手写终极版LLM-MicroCortex)
11. [手写MicroCortex的SFT](./11.手写MicroCortex的SFT)
12. [手写MicroCortex的LoRA](./12.手写MicroCortex的LoRA)
13. [手写MicroCortex的DPO](./13.手写MicroCortex的DPO)
14. [手写PPO](./14.手写PPO)
15. [多模态大模型前置知识](./15.多模态大模型前置知识)
16. [手写终极版VLM-MicroCortex-V](./16.手写终极版VLM-MicroCortex-V)
17. [大模型评价指标](./17.大模型评价指标)
18. [基于多模态大模型MicroCortex-V的六轴机械臂具身智能](./18.基于多模态大模型MicroCortex-V的六轴机械臂具身智能)
19. [手写基于langchain智能体](./19.手写基于langchain智能体)
20. [MCP前置知识](./20.MCP前置知识)