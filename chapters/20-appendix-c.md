# 附录C 原始资料阅读路线

## C.1 先读结构，再读训练和系统

本书以原始论文、作者维护的代码与官方模型资料为主要事实依据。以下是便于复查的主题入口；各章相邻段落还给出了更精确的链接。查询日期为2026年10月23日。研究发布日期与网站最近抓取日期不同，阅读时不要把后者误当成模型发布年份。

1. [Attention Is All You Need，2017](https://arxiv.org/abs/1706.03762)：先看架构和缩放点积定义，再检查编码器与解码器的信息可见性。
2. [Neural Machine Translation of Rare Words with Subword Units，2016](https://aclanthology.org/P16-1162/)：理解子词单位为何能处理罕见词；不要将其直接等同于GPT-2全部tokenizer细节。
3. [Layer Normalization，2016](https://arxiv.org/abs/1607.06450)：明确统计量沿哪些维度计算。
4. [GPT-2技术报告，2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)与[官方代码](https://github.com/openai/gpt-2)：结合源代码核对词表、位置表、Pre-LN与参数计数。
5. [RoFormer，2021](https://arxiv.org/abs/2104.09864)：先跟踪成对坐标的旋转，再看相对位置信息怎样进入点积。
6. [RMSNorm，2019](https://arxiv.org/abs/1910.07467)：对照LayerNorm，分清是否中心化。
7. [GLU Variants Improve Transformer，2020](https://arxiv.org/abs/2002.05202)：关注门控网络的矩阵数量与参数预算。
8. [FlashAttention，2022](https://arxiv.org/abs/2205.14135)：重点理解IO组织和在线softmax，而不是将它误读成稀疏注意力。
9. [GQA，2023](https://arxiv.org/abs/2305.13245)：检查查询头与KV头的共享关系。
10. [LoRA，2021](https://arxiv.org/abs/2106.09685)：区分低秩增量微调与MLA主体中的低秩瓶颈。
11. [Direct Preference Optimization，2023](https://arxiv.org/abs/2305.18290)：关注偏好对、参考模型和序列log概率。

## C.2 DeepSeek：区分结构与后训练

12. [DeepSeekMoE，2024](https://arxiv.org/abs/2401.06066)：阅读细粒度专家与共享专家的组织方法。
13. [DeepSeek-V2，2024](https://arxiv.org/abs/2405.04434)：跟踪MLA内容分支、位置分支与KV缓存。
14. [DeepSeek-V3技术报告，2024](https://arxiv.org/abs/2412.19437)与[官方仓库](https://github.com/deepseek-ai/DeepSeek-V3)：将配置数字与推理实现对应起来。
15. [DeepSeekMath，2024](https://arxiv.org/abs/2402.03300)：理解GRPO组内比较的来源与用途。
16. [DeepSeek-R1，2025](https://arxiv.org/abs/2501.12948)与[官方仓库](https://github.com/deepseek-ai/DeepSeek-R1)：分别检查R1-Zero、R1和distill模型，别把名称继承误解成架构继承。

## C.3 视觉、全模态与动作

17. [Vision Transformer，2020](https://arxiv.org/abs/2010.11929)：关注patch如何成为序列以及位置表示。
18. [CLIP，2021](https://arxiv.org/abs/2103.00020)：先把批内图文相似矩阵手算一遍，再理解对比目标。
19. [Visual Instruction Tuning / LLaVA，2023](https://arxiv.org/abs/2304.08485)：检查视觉投影与指令数据的分工。
20. [Qwen2.5-VL技术报告，2025](https://arxiv.org/abs/2502.13923)：关注动态分辨率、视觉位置和视频时间坐标。
21. [GPT-4o System Card，2024](https://arxiv.org/abs/2410.21276)与[具体API模型资料](https://developers.openai.com/api/docs/models/gpt-4o)：研究能力与端点支持范围需要分别引用，未披露的内部参数不要补猜。
22. [RT-2，2023](https://arxiv.org/abs/2307.15818)：观察动作如何接入视觉语言建模。
23. [OpenVLA，2024](https://arxiv.org/abs/2406.09246)与[官方项目](https://openvla.github.io/)：跟踪视觉编码、语言主干与动作离散化。
24. [π0，2024](https://arxiv.org/abs/2410.24164)与[openpi代码](https://github.com/Physical-Intelligence/openpi)：比较连续动作块的训练目标与采样实现，特别留意时间方向约定。

## C.4 四维算子的官方接口依据

以下接口资料于2026年10月26日核对，链接固定到PyTorch 2.14文档；书中的小整数、分镜与训练样例均由本书独立构造。

25. [torch.permute](https://docs.pytorch.org/docs/2.14/generated/torch.permute.html)：轴置换返回原张量的视图；结合第01章索引和stride表阅读。
26. [torch.matmul](https://docs.pytorch.org/docs/2.14/generated/torch.matmul.html)：最后两根轴参加矩阵计算，前面的轴按广播规则匹配；对应第03、12章。
27. [torch.nn.Conv2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html)：核对NCHW输入、输出、卷积核及偏置的轴定义；对应第12、17章。

## C.5 怎样读出一篇新论文的结构

先写出输入和输出，再画每个中间表示。对每个“高效”声明分别问：减少参数、减少实际计算、减少常驻内存，还是减少数据搬运？对每个“能力提升”声明分别问：训练数据、模型结构、后训练、推理预算是否同时改变？对每个“开放”声明分别问：代码、权重、训练数据、许可和复现材料各开放了什么？

然后把文章中的一个关键机制缩小到可运行算例。若是新注意力，写出四个位置的小矩阵；若是新路由，画出token到专家的分发表；若是新模态，列出采样率、切块规则与接口长度；若是新训练目标，用三条或四条样本手算一次。这个习惯可以把今天读懂的模型，转化成以后阅读未知架构的能力。
