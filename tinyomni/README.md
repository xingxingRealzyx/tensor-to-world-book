# TinyOmni：从随机权重训练的有限世界多模态模型

TinyOmni 是本书的可运行综合实验：352,281 个默认参数，自己生成数据，真实计算损失、反向传播并执行 AdamW 更新。文本、16×16 图像、256 点合成声音、四帧视频经过不同入口进入共享 Transformer，再生成文本、规范图像、规范声音和适用时的二维动作轨迹。推理加载真实检查点，不调用外部模型或网络服务。

它学习的是 **3 种颜色 × 3 种形状 × 2 种运动方向** 的人工世界。声音用频率编码颜色、振幅编码形状，是合成音调而非人类语音；文字是固定的小词表，而非任意自然语言。它能演示“文字→图像/声音”“图像→文字/声音”“声音→文字/图像”这些计算路径，但不具备真实照片理解、自然语音识别或开放领域生成能力。输入都有随机扰动，输出是规范目标，因此不是把带噪输入直接复制到输出。

## 安装与设备

从书籍根目录 `tensor-to-world-book` 运行以下命令：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r tinyomni/requirements.txt
python -c "import torch,numpy as np; print(torch.__version__); print(torch.from_numpy(np.zeros(1)).numpy()); print('CUDA available:', torch.cuda.is_available())"
```

用户目标硬件为 RTX 3090 24 GB。请使用与本机驱动相容的 CUDA 版 PyTorch；上面的检查必须显示 CUDA 可用，才能用 `--device cuda`。本项目不要求租用 GPU。`--device auto` 优先 CUDA，其次 MPS，最后 CPU；在没有 NVIDIA GPU 的本机实测中显式使用 `--device cpu`。本文不把 CPU 结果当成 RTX 3090 的性能测量。

PNG、WAV、JSON 导出以及 HTTP 服务使用 Python 标准库；`custom_infer` 的用户图片读取使用 Pillow，已列入依赖。没有音频驱动、数据库或云端 API 依赖。

上面的激活命令适用于 macOS/Linux；Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`。本版实际验证环境为 Python 3.12.14、PyTorch 2.14.1、NumPy 2.5.4、Pillow 13.0.0；完整记录在根目录 `validation-report.json`。GPU 用户应按 PyTorch 官方安装选择器安装 CUDA 构建后再安装本项目依赖。

建议使用 Python 3.12 与当前官方 CUDA PyTorch。探针还检查 NumPy 与 PyTorch 的数组互转；若旧 PyTorch wheel 与 NumPy 2.x 组合出现 `_ARRAY_API` 或 `Numpy is not available`，先更新到兼容构建，再运行训练。依赖文件的宽版本范围不是全部历史版本的测试矩阵。

## 从零训练

```bash
python -m tinyomni.train \
  --tasks text,image,audio,video,mixed \
  --steps 1000 --batch-size 32 --device auto \
  --seed 42 --eval-samples 128 --out output/tinyomni
```

CPU 快速实测命令：

```bash
python -m tinyomni.train \
  --steps 100 --batch-size 32 --device cpu \
  --seed 42 --eval-samples 64 --out output/tinyomni-cpu100
```

默认结构为宽度 64、4 个头、前馈宽度 128、2 层编码器与 2 层解码器，dropout=0。总损失为文本下一 token 交叉熵，加图像、声音、动作的三个掩码 MSE；各 MSE 先对单样本所有输出元素取均值，再在有效样本间平均。动作不可从观察推知时，不计算动作损失。不同任务的数据在训练中混合，优化器看到真实不断变化的批次。

输出目录包含：

- `checkpoint.pt`：模型权重、配置、AdamW 状态、更新步数、训练记录与词表。
- `config.json`：架构、参数数、设备和训练设置。
- `metrics.json`：每一步真实损失、训练前与训练后的分任务验证、模态消融结果。
- `loss-curve.svg`：直接由真实逐步损失绘制的曲线。

可用 `--eval-every 100` 保存中途验证和检查点。`--log-every 25` 控制终端日志频率，所有步骤的损失仍会写入 JSON。出现非有限损失或梯度时停止更新，不把 NaN 继续写进模型。

继续训练会恢复权重和优化器状态：

```bash
python -m tinyomni.train \
  --resume output/tinyomni/checkpoint.pt \
  --steps 1500 --batch-size 32 --device auto \
  --seed 42 --out output/tinyomni-resumed
```

`--steps` 是最终总步数，以上从 1000 步继续到 1500 步。恢复时要求原来的 seed 与 tasks；保存的 AdamW 学习率和权重衰减优先于命令行新值。没有优化器状态的文件会报错，绝不静默改成“只恢复权重”。跨设备、不同 PyTorch 版本或改变批大小不保证逐比特相同。

## 如何判断是否真的学会使用模态

验证执行 `model.generate(...)`，逐 token 自回归产生答案，不使用 teacher forcing 的预测来冒充自由生成结果。文本目标依次为颜色、形状、运动、EOS；PAD 被忽略，UNK 是必须正确预测的目标。静态图片和声音没有运动信息，所以正确运动答案是 UNK；它们的动作 MSE 为 JSON `null`，并有 `action_samples=0`，不能当成零误差成绩。

每个任务单独报告 token accuracy、四 token 全部正确的 exact match，以及图像/声音/动作 MSE 和有效样本数。训练种子限定在 `[0,2^63)`，验证种子限定在 `[2^63,2^64)`；验证端使用能够利用完整 64 位种子的独立随机数生成路径。低位训练仍采用原来的 `seed+step` 序列，所以 seed=42 的既有训练权重不受影响。这是相同有限属性组合下的随机位置、相位和噪声划分，**不是未见属性组合泛化实验**。只看包含固定 EOS、UNK 的 token accuracy 容易高估能力，因此必须同时看 exact match 和连续输出。

还会为 image 与 audio 各做一次成对消融：同一组样本先正常生成，再将对应传感器张量置零并令 presence=False。若成绩几乎不变，应怀疑模型忽略该通道。消融后仍可能答中常见颜色、形状，EOS 和 UNK 也容易预测；零输入成绩不应该期待必然为零。连续头只使用条件编码器，既不读取真实标签，也不读取 teacher-forced 解码器输入。

随书检查点 `runs/tinyomni-final/checkpoint.pt` 在本机 CPU 上从零训练 1000 步，batch=32、seed=42。最终验证每个任务使用 256 个样本，训练权重保持不变，验证采用上述独立高位随机流。完整损失和评测位于同目录的 `metrics.json`：

| 项目 | 结果 |
|---|---:|
| 第1步 → 第1000步总损失 | 5.4214 → 0.02330 |
| text / image exact match | 100% / 98.44% |
| audio / video / mixed exact match | 97.27% / 98.44% / 97.66% |
| 成对图像消融 exact match | 96.88% → 10.94% |
| 成对声音消融 exact match | 98.05% → 9.38% |

首末损失来自不同随机批次，不能代替验证指标。消融使用另一组固定验证样本，所以其正常成绩与主验证表不完全相同。这些数值只衡量有限合成世界中的随机扰动泛化，不衡量真实图像、自然语音或未见属性组合的能力。

旧检查点无需重训就能按新的验证种子域重新评估：

```bash
python -m tinyomni.train --evaluate-only \
  --resume output/tinyomni/checkpoint.pt --out output/tinyomni \
  --seed 42 --eval-samples 128 --batch-size 32 --device auto
```

该命令不执行优化器更新，也不改写 `checkpoint.pt`；它更新 `metrics.json`、`config.json` 与已有损失曲线，旧验证记录保留在 `prior_evaluations` 中，新的 `final` 记录明确带验证种子域。`--steps` 在此模式下不参与训练。

## 本地推理并保存产物

```bash
python -m tinyomni.infer \
  --checkpoint output/tinyomni/checkpoint.pt \
  --task audio --seed 12345 --device auto \
  --out output/tinyomni-example
```

`--task` 可取 `text,image,audio,video,mixed`。该命令生成一个新的可复现合成样本，保存 `predicted.txt`、`prediction.json`、输入/生成/目标 PNG 与 WAV、四帧输入视频 PNG、`actions.json`。目标文件只是便于人工对照的合成真值，不会送进模型。缺失模态的输入文件为零，与 presence 标记对应。

波形原长只有 256 点；导出采用 8,000 Hz，时长 32 毫秒。带 `_repeated_preview.wav` 的文件只是把同一短输出重复 20 次方便试听，不是模型产生了更长语音。归一化动作是 `[4,2]` 的二维教学轨迹，不是任何机械臂的控制命令。动作目标不可知时，网络内部虽有数值输出，但该部分没有监督，接口不把它作为有效动作暴露：JSON 中 `action_available=false`、`predicted_actions=null`。

## 输入自己的受控文字、图片或声音

先使用随书权重验证完整路径：

```bash
python -m tinyomni.custom_infer --checkpoint runs/tinyomni-final/checkpoint.pt --text "blue square left" --device cpu --out runs/my-custom-text
python -m tinyomni.custom_infer --checkpoint runs/tinyomni-final/checkpoint.pt --image runs/tinyomni-final/demo-image/input_image.png --device cpu --out runs/my-custom-image
python -m tinyomni.custom_infer --checkpoint runs/tinyomni-final/checkpoint.pt --audio runs/tinyomni-final/demo-audio/input_audio.wav --device cpu --out runs/my-custom-audio
```

这里的文件直接作为模型输入，不会重新按 seed 生成。文字语法为颜色、形状、方向三个词；图片必须为 16×16，内容应符合训练中的彩色图形协议；声音必须是 8,000 Hz、单声道 PCM16、恰好 256 个采样，并符合训练的频率/振幅协议。文件格式正确并不意味着任意照片或录音属于模型学过的分布。模型输出会保存为 PNG、WAV、文本与 JSON；由于没有标签文件，这些输入不会报告准确率。

## 本地 HTTP 部署

```bash
python -m tinyomni.serve \
  --checkpoint output/tinyomni/checkpoint.pt \
  --device auto --port 8765
```

服务只绑定 `127.0.0.1`。另开终端执行：

```bash
curl http://127.0.0.1:8765/health
curl -X POST http://127.0.0.1:8765/infer \
  -H 'Content-Type: application/json' \
  -d '{"task":"image","seed":12345}'
```

`GET /health` 返回已加载模型的状态。`POST /infer` 只接受 `task` 与整数 `seed`，现场创建合成输入并进行真实模型推理，返回预测文本、动作以及生成 PNG/WAV 的 Base64 数据。它没有文件上传、任意照片识别或自然语音接口。请求大小限制为 4 KiB，读取默认超时 5 秒（可用 `--read-timeout` 调整），不完整请求体超时返回 408。HTTP 请求由不同线程处理，推理通过锁串行访问一个模型实例，所以未完成的上传不会占住健康检查；它仍不是面向公网的生产服务器。

## 验证代码本身

```bash
python -m tinyomni.tests
```

测试覆盖真实参数更新、非有限损失拒绝更新、检查点与优化器恢复、解码器未来位置不能影响较早 logits、连续头和生成对训练标签独立、不可观察动作的掩码、PNG/WAV 导出、训练/验证随机流分离，以及 HTTP 半包超时后健康检查恢复。只评估模式还会检查权重文件没有被改写。测试通过验证接口与训练链路；任务能力仍以独立随机样本上的实际训练结果为准。
