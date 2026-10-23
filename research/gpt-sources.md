# 第05—08章：原始资料、算例核验与配图规格

核对日期：2026-10-23。正文讨论固定历史架构与基本算法，不引用实时排行榜。下列链接均为原始论文、论文作者机构或官方源代码；检索时出现的博客和百科没有作为事实依据。正文中的教学概率、向量、矩阵和假想缓存配置均明确标注为构造算例，不是训练或硬件性能实测。

## 一、原始资料清单

| 资料 | 年份 | 本书使用范围 |
|---|---:|---|
| [Radford 等：Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language-models.pdf) | 2019 | GPT-2 背景、byte-level BPE、预训练目标与模型规格 |
| [OpenAI GPT-2：model.py](https://github.com/openai/gpt-2/blob/master/src/model.py) | 2019 | Pre-LN、两次残差、最终 LN、共享输出头、K/V past |
| [OpenAI GPT-2：encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py) | 2019 | 可逆字节映射、正则预切分、BPE 优先级 |
| [OpenAI GPT-2：README 参数统计更正](https://github.com/openai/gpt-2) | 2019 | 117M 与发布配置 124M 的区别 |
| [Lee 等：Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) | 2021/2022 | 重复训练数据、训练测试重叠与评估可信度 |
| [Loshchilov、Hutter：Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | 2017/2019 | AdamW 与独立权重衰减 |
| [Ouyang 等：Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) | 2022 | SFT、示范数据与偏好训练阶段的区别 |
| [Rafailov 等：Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | 2023 | DPO 的参考模型与对数概率比目标 |
| [Holtzman 等：The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) | 2019/2020 | nucleus / top-p sampling 与生成策略 |
| [Ainslie 等：GQA](https://arxiv.org/abs/2305.13245) | 2023 | MHA/MQA/GQA 的 KV 头共享 |
| [Frantar 等：GPTQ](https://arxiv.org/abs/2210.17323) | 2022/2023 | 训练后权重量化的代表方法 |
| [Lewis 等：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | 2020 | 检索与生成结合；正文区分原始方法与广义工程用法 |
| [Hu 等：LoRA](https://arxiv.org/abs/2106.09685) | 2021 | 低秩参数增量与参数量公式 |
| [Su 等：RoFormer](https://arxiv.org/abs/2104.09864) | 2021 | 二维旋转位置编码与相对位置点积 |
| [Zhang、Sennrich：Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | 2019 | RMSNorm 与 LayerNorm 的差别 |
| [Shazeer：GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | 2020 | 门控前馈与 SwiGLU |
| [Touvron 等：LLaMA](https://arxiv.org/abs/2302.13971) | 2023 | 常见现代解码器组件的历史实例 |
| [Meta 官方 Llama：model.py](https://github.com/meta-llama/llama/blob/main/llama/model.py) | 官方仓库 | RMSNorm、RoPE、GQA、SwiGLU 的实现核对；代码当前属于 Llama 2 系列实现，不称作 LLaMA 1 的固定代码 |
| [Chen 等：Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) | 2023 | 位置插值与上下文扩展的边界 |
| [Liu 等：Lost in the Middle](https://arxiv.org/abs/2307.03172) | 2023 | 长上下文使用能力与输入窗口上限的区别 |
| [Dao 等：FlashAttention](https://arxiv.org/abs/2205.14135) | 2022 | 分块、在线 softmax、IO 感知与复杂度边界 |

## 二、已核对的官方实现细节

- GPT-2 `default_hparams`：上下文 1024、宽度 768、12 个头、12 层。
- GPT-2 `block`：`attn(norm(x))` 后残差相加，再 `mlp(norm(x))` 后残差相加。
- GPT-2 `mlp`：宽度扩展为 4D，GELU，再线性投影回 D。
- GPT-2 `model`：token 与位置查表相加；末尾独立 `ln_f`；输出 logits 乘 `wte` 的转置，没有新增独立输出权重。
- GPT-2 官方 README 明确承认早期 small 117M、medium 345M 计数有误。正文没有编造“117M 是另一种合法统计口径”。
- Meta 官方代码 RMSNorm 使用均方根、缩放向量；SwiGLU 是 `w2(silu(w1(x))*w3(x))`；FFN 宽度先作 `2/3` 调整再按硬件倍数取整；RoPE 用在 Q/K；缓存 K/V。
- 原始 FlashAttention 是 IO 感知的精确密集注意力执行方案；正文未把它表述成线性时间注意力。

## 三、独立数值核验

所有下述数值由标准 Python 浮点与整数运算独立核对。没有另写永久测试脚本，以免越出负责文件边界；本清单记录可重算的输入及结果。

### GPT-2 参数账本

```text
D=768, V=50257, T_max=1024, L=12
token = 50257*768 = 38597376
position = 1024*768 = 786432
qkv = 768*2304 + 2304 = 1771776
attention_output = 768*768 + 768 = 590592
mlp_up = 768*3072 + 3072 = 2362368
mlp_down = 3072*768 + 768 = 2360064
two_layernorms = 4*768 = 3072
one_block = 7087872
all_blocks = 85054464
final_layernorm = 1536
total = 124439808
```

共享输出头不重复计数。不包含激活、梯度、优化器状态、KV cache。

### 损失、采样与 DPO

```text
correct_probs = [0.8,0.5,0.25]
mean(-ln(p)) = 0.7675283643313485
perplexity = 2.1544346900318834

logits = [2,1,0,-1]
tau=0.5 -> [0.864955,0.117059,0.015842,0.002144]
tau=1.0 -> [0.643914,0.236883,0.087144,0.032059]
tau=2.0 -> [0.455054,0.276004,0.167405,0.101536]
tau=1.0, top-p=0.8 -> retain first two -> [0.731059,0.268941]

DPO reference pair probabilities = [0.08,0.08]
DPO current pair probabilities = [0.12,0.04]
beta = 0.1
unscaled log-ratio difference = ln(3)
loss = ln(1+exp(-0.1*ln(3))) = 0.6397244942261298
```

DPO 两个候选概率不需要加和为 1，参考与当前概率只是完整回答空间的两个条目。

### 缓存容量与 LoRA

```text
KV_bytes = 2*L*B*T*H_kv*d_h*bytes
L=32, B=1, T=8192, d_h=128, bytes=2
H_kv=32 -> 4294967296 bytes = 4 GiB
H_kv=8  -> 1073741824 bytes = 1 GiB
H_kv=1  -> 134217728 bytes = 128 MiB

LoRA D_in=D_out=4096, r=8
full_weight = 4096*4096 = 16777216
low_rank_update = 8*(4096+4096) = 65536
```

### SwiGLU 与 FlashAttention 行级示例

```text
D=768, M=2048:
3*D*M = 8*D*D = 4718592

SwiGLU x=[1,2]
xW_gate=[1,2,1]
SiLU = [0.7310586,1.7615942,0.7310586]
xW_up=[3,1,2]
gated=[2.1931757,1.7615942,1.4621172]
output=[3.6552929,0.2994770]

Flash row scores=[1,2,3,4], values=[10,20,30,40]
first block: m=2, l=1.3678794, u=23.6787944
second block: m=4, l=1.5530018, u=54.2409596
output=u/l=34.9265273
```

RoPE 例中单位步长为 60°，明确属于教学角度。真实经典频率使用 `base^(-2i/d_h)` 的弧度频率，不把 60° 写成模型配置。

## 四、交给主编的配图规格

### figures/05-gpt2.svg

图名：GPT-2 完整前向路径。主线 `ids[B,T] → E+P[B,T,768] → 12×Pre-LN block → final LayerNorm → @E^T → logits[B,T,50257]`。给 token 表 E 标 `[50257,768]`，位置表 P 标 `[1024,768]`。从 E 画一条共享权重连线到输出头，避免让读者以为两张独立表。展开一个 block：X 的旁路绕过 LN1+Attention 后与注意力输出相加为 Y；Y 的旁路绕过 LN2+MLP 后相加为 Z。两条残差在 LN 前分叉；不能画成归一化后的值构成直通残差。

### figures/06-training.svg

图名：标签移位与并行训练。用完整序列 `a,b,c,d`，输入行为 `a,b,c`，目标行为 `b,c,d`，三条垂直预测箭头。下方展示 3×3 下三角可见矩阵：第一行仅第一格可见，第二行前两格，第三行三格。标注“同一次前向并行预测三个位置”“目标是当前位置的下一个 token”。不要把训练画成先生成 b 再生成 c 的串行推理过程。

### figures/07-kv.svg

图名：prefill 与 decode 的缓存路径。左侧 prompt 作为多行 `[B,T_prompt,D]` 输入，经过若干层，为每层写入 K/V；最后位置 logits 选择第一个输出 token。右侧将该 token 作为 `[B,1,D]` 输入，产生新 Q/K/V，追加 K/V 一行，再让 Q 查询旧缓存加新行。明确 Q `[B,H,1,d_h]`，缓存 K/V `[B,H_kv,T,d_h]`。底部容量对比：固定 L32、B1、T8192、d_h128、每元素2字节，32 KV 头4GiB，8 KV 头1GiB。标注“理论缓存，不含权重与其他开销”。

### figures/08-rope.svg

图名：共同位置平移保留相对旋转。两幅单位圆。左图未旋转 q=k=(1,0)，q 在位置1旋到60°，坐标(0.5,0.866)；k 在位置3旋到180°，坐标(-1,0)；相对角120°、点积-0.5。右图两个位置都加1，q在120°、k在240°，相对角仍120°、点积仍-0.5。所有角度按逆时针画，不要把位置3的k误画成120°。页脚“每步60°为教学设定；真实模型每个二维组使用其配置频率”。

## 五、交稿检查

- 05—08 四章共约 13,800 个汉字，另有公式、英文术语、表格与参考链接；每章均为完整连续书稿。
- 每章三道练习及简答；易错点分布在专节或紧邻相关概念处。
- 所有表格不超过四列；公式使用纯文本 code fence，无 LaTeX 宏。
- 正文图引用按主编要求使用 `figures/*.svg`。
- 仅编辑指定四章与本研究记录；未修改其他章节、配图或排版文件。
