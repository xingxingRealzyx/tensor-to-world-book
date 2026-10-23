# 第12—16章一手来源与核对记录

核对日期：2026-10-23。正文中的教学张量、数值矩阵、模拟器项目和评测方案为本书自行构造，不能视为下列模型的真实配置。全部引用采用直接 Markdown 链接，未把检索摘要作为唯一证据。

## 第12章：视觉编码与图文对比

- ViT，An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale，2020/2021：https://arxiv.org/abs/2010.11929 。支持图像分块、线性投影、位置表示与分类 token 的基础设计。书中 224/16、196、768 以及扩大分辨率后的复杂度为独立算术推演。
- CLIP，Learning Transferable Visual Models From Natural Language Supervision，2021：https://arxiv.org/abs/2103.00020 。支持双编码器、图文对比训练、文本候选的零样本迁移。官方代码：https://github.com/openai/CLIP 。书中三乘三 logits 是教学构造，不是实测相似度；双向 softmax 的数值可手算验证。

## 第13章：视觉语言连接与动态输入

- LLaVA，Visual Instruction Tuning，2023：https://arxiv.org/abs/2304.08485 。支持视觉编码器连接 LLM 和视觉指令训练的基本范式。
- LLaVA-1.5，Improved Baselines with Visual Instruction Tuning，2023/2024：https://arxiv.org/abs/2310.03744 。用于区分早期线性连接器与后续 MLP 连接器等版本差异。正文不混写各版本的视觉分辨率和隐藏维度。
- Qwen2.5-VL Technical Report，2025：https://arxiv.org/html/2502.13923v1 。第2.1节明确：视觉 patch 步长 14，输入尺寸处理为 28 的倍数，四个空间邻接特征合并后通过两层 MLP；多数层窗口注意力；视频动态帧率及对齐绝对时间的 MRoPE。正文只摘取机制，不重复论文排行结论。448×448 的 1024→256 是空间计数示意，已说明真实处理器还管理时间与预算。

## 第14章：语音、视频与 GPT-4o 边界

- Whisper，Robust Speech Recognition via Large-Scale Weak Supervision，2022：https://arxiv.org/abs/2212.04356 。用作音频特征编码、文本解码的公开范例；窗长400、步长160、10秒998帧是显式无填充教学计算，不宣称等同 Whisper 完整处理流程。
- EnCodec，High Fidelity Neural Audio Compression，2022：https://arxiv.org/abs/2210.13438 。用作神经音频压缩与量化表示的公开实例；50位置/秒、8码本是教学选择，不冒称其唯一参数。
- GPT-4o 2024 发布说明：https://openai.com/index/hello-gpt-4o/ 。已实际打开，支持跨文本、视觉、音频端到端训练的官方公开表述及历史能力背景。未根据示例推断非公开层数、参数、专家、tokenizer、codec或路由。
- GPT-4o API 模型页：https://developers.openai.com/api/docs/models/gpt-4o 。已实际打开，在核对日模态表写明文字输入输出、图像仅输入，音频与视频不支持。正文严格把普通模型ID端点能力与历史omni研究介绍分开，不把产品能力作为完整内部架构证据。

## 第15章：动作离散化、连续块与闭环

- RT-2，Vision-Language-Action Models Transfer Web Knowledge to Robotic Control，2023：https://arxiv.org/abs/2307.15818 。支持动作文本 token 与视觉语言/轨迹联合微调。项目：https://robotics-transformer2.github.io/ 。正文避免把网页预训练等同物理操纵训练。
- OpenVLA，An Open-Source Vision-Language-Action Model，2024：https://arxiv.org/html/2406.09246v3 。支持 DINOv2/SigLIP 视觉特征和逐动作维256区间量化、动作下一token训练。官方实现：https://github.com/openvla/openvla 。正文区分原始 OpenVLA 与后续动作块优化。
- π0，A Vision-Language-Action Flow Model for General Robot Control，2024：https://arxiv.org/html/2410.24164v1 。支持 PaliGemma 初始化、单独 action expert、连续动作块、flow matching。正文不把 action expert 等同于语言模型动态 top-k MoE。公开代码入口：https://github.com/Physical-Intelligence/openpi 。2×2矩阵和[B,K,7]均为教学例，流生成进度与机器人物理时间明确分开。
- Open X-Embodiment，Robotic Learning Datasets and RT-X Models：https://robotics-transformer-x.github.io/ 。支持跨数据来源、跨机器人形态的数据整合背景。正文不引用可能随时间变化的数据规模与机器人数量。

## 第16章：自洽项目与评测

本章是书中概念的自主综合设计，不声称复现任何现有系统。80亿参数×2字节=160亿字节为十进制16GB，只代表权重存储。441→147的序列长度及9倍位置对差异是教学推演，总延迟预算是明确标注的假设。

## 原创插图规格

- `12-patches.svg`：224×224×3 图片，14×14网格，196×768展平矩阵，投影为196×D；可选CLS须单独标为197。
- `12-clip.svg`：行是猫/车/苹果图片，列是配对文本；logits为[[3,1,0],[0,3,1],[1,0,3]]；对角线高亮，行和列双向监督。
- `13-vlm.svg`：教学拼接路径[B,196,768]→[B,196,2048]与[B,32,2048]拼成[B,228,2048]；另示文本Q与视觉KV的交叉注意力。注明通用示意。
- `14-omni.svg`：ASR→文字LLM→TTS 与联合声/图/文处理并列，突出文本瓶颈与时间对齐；注明并非GPT-4o内部架构。
- `15-vla.svg`：观察→VLA→[B,K,7]→仅执行前r步→新观察回环；旁示噪声动作矩阵经若干更新成动作块，区分生成进度s和动作时间k。
- `16-system.svg`：采集/对齐→编码→决策→输出/动作→验证/反馈；接口旁标来源、schema、时间戳、单位与版本。
