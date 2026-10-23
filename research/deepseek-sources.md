# 第09—11章来源与核验记录

核验日期：2026-10-23。章节固定讨论 DeepSeek-V2、原始 DeepSeek-V3 与原始 DeepSeek-R1，不声明它们是当前最新型号。论文链接指定版本；仓库 main 代表核验时公开材料，正式出版时宜进一步保存对应提交快照。

## 一手来源

| 来源 | 书稿用途 | 版本边界 |
|---|---|---|
| [DeepSeek-V3 报告](https://arxiv.org/html/2412.19437v2) | 参数规模、MoE 平衡机制、层配置 | arXiv v2，原始 V3 |
| [DeepSeek-V3 官方模型配置](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) | 实际维度、专家数量、路由缩放 | deepseek-ai 官方模型库 |
| [DeepSeek-V3 官方推理代码](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py) | naive 与 absorb 两条计算路径、缓存 Norm 后潜变量 | 核验时 main |
| [DeepSeek-V2 报告](https://arxiv.org/html/2405.04434v5) | MLA、矩阵吸收、解耦 RoPE | arXiv v5，V2 |
| [GQA 原始论文](https://arxiv.org/abs/2305.13245v3) | GQA 与 MLA 的结构区别 | arXiv v3 |
| [R1 原始报告](https://arxiv.org/html/2501.12948v1) | Zero、R1 两条路线及训练阶段 | arXiv v1；不混入 R1-0528 |
| [R1 官方仓库](https://github.com/deepseek-ai/DeepSeek-R1) | 基础模型、稠密蒸馏模型列表、许可说明 | 核验时 main |
| [DeepSeekMath 报告](https://arxiv.org/html/2402.03300v3) | GRPO 的 token 级目标与结果奖励优势 | arXiv v3 |
| [V3 代码许可证](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-CODE) | 代码 MIT 与模型许可的区分 | 核验时 main |
| [V3 模型许可证](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL) | 模型许可为单列工件，不能等同代码许可 | 核验时 main |

章节的矩阵与概率数值例子均为原创教学构造，不复制论文实验数据。论文提供架构与方法依据，toy 示例不宣称可以重现论文能力结果。

## 编辑时应保留的区分

- 37B 是每 token 的激活参数口径，不是模型全部常驻权重，也不是推理显存。671B 的 BF16 权重估算只用于量级教学。
- V3 的选路偏置只影响选择；合并权重来自原始亲和分数。归一化后还可有整体缩放。书中 toy 缩放取 1。
- 主要策略称 auxiliary-loss-free，不等于报告中完全不存在辅助损失。
- GQA 分享实际 K、V 头，MLA 分享联合潜变量，两者不是同义词。
- MLA 的吸收等价从固定 C 出发，不可把潜变量上的非线性 Norm 随意折叠到线性权重里。
- R1-Zero 已有预训练基础；原始 R1-Distill 的学生是稠密 Qwen/Llama 系列，未继承 V3 MoE/MLA。
- GRPO 手算明确使用总体标准差。真实实现的标准差定义、长度处理和稳定项需要逐项核对。
- 可验证终值并不验证所有中间推理；pass@k 不等于无验证器的单次交付正确率。

## 数例复核

09：路由矩阵每行之和为 1；top-2 共 8 次分配；三专家负载为 3、2、3。以 E1(x)=x、E2(x)=2x、E3(x)=-x、Shared(x)=0.5x，输出 F 为 [[11/6,11/3],[11/3,0],[0,0],[0,11/9]]。

10：B=1、T=8192、L=32、h=32、dk=dv=128、2 字节元素，MHA=4 GiB；改 8 个 KV 头为 1 GiB。V3 大小示例 B=1、T=32768、L=61，潜缓存为 2,302,672,896 字节，即 2.14453125 GiB。展开内容 KV 但位置键仅存一份为 122.23828125 GiB；位置键重复物化到各头为 152.5 GiB。这是缓存布局计算，不包含权重、分页开销、工作区或并行复制。

10：toy 注意力 logits 为 [1,6]/sqrt(2)，softmax=[0.0283179185,0.9716820815]；展开与吸收路径输出均为 [0.0566358371,1.9433641629]。

11：奖励 [1,0,0,0] 的均值 0.25，总体方差 0.1875，优势 [sqrt(3),-1/sqrt(3),-1/sqrt(3),-1/sqrt(3)]。概率例子 1-0.7^4=0.7599，明确依赖独立尝试假设。

## 插图精确说明

09-moe.svg：四个 token 进入 4×3 路由热图，行依次 [2/3,1/3,0]、[0,7/9,2/9]、[1/4,0,3/4]、[5/9,0,4/9]。E1 获 token 1/3/4，E2 获 1/2，E3 获 2/3/4。共享专家旁路接收全部 token。输出按 token 原位置合并，残差若绘制必须与专家增量 F 分开标记。

10-mla.svg：X 分成内容潜变量 C[B,T,r] 与旋转后位置键 K_R[B,T,p]，二者进入缓存。上投影到展开 K_C/V 的路径用虚线；实线展示 q_C @ U_K^T 得 q_latent，再与 C 点积；位置分数与内容分数相加，softmax 后读取 C 并做值上投影。标记“缓存 r+p，不是每头完整 KV”。

11-grpo.svg：同题生成四条候选，验证器输出 [1,0,0,0]，组均值 0.25、总体标准差约 0.433，优势 [1.732,-0.577,-0.577,-0.577] 进入策略更新。把“更新权重”画在训练环，固定权重推理不画回传箭头。
