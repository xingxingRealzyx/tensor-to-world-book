# 附录A 实验导航与可复核的方法

## A.1 随书工程怎样组织

正文、图解、练习和答案均完整排入可打印的PDF。配套源码中，`chapters/`保存章节，`labs/`保存逐章实训，合并后的`book.md`包含完整内容；`figures/`保存矢量插图，HTML副本也内嵌了这些静态插图。先用书中的小矩阵和纸笔核对原理，再运行本附录列出的训练与部署命令。

| 文件 | 用途 | 何时阅读 |
| --- | --- | --- |
| code/mini_transformer.py | NumPy前向与KV缓存 | 第03-07章 |
| code/verify_examples.py | 手算数字自动复核 | 第01-12章 |
| tinyomni/lab.py | 第01-11章独立小训练 | 对应章节结束时 |
| tinyomni/dataset.py | 合成世界与监督边界 | 第12章起 |
| tinyomni/model.py | 共享模型与模态接口 | 第13-17章 |
| tinyomni/train.py | 优化器、检查点与评测 | 第06与17章 |
| tinyomni/infer.py、serve.py | 加载与本地部署 | 第07与17章 |
| tinyomni/custom_infer.py | 显式条件与文件推理 | 跑通最终项目后 |

第一遍无需逐行读完所有实现。先运行一章实验，再在源码中找到与本章公式对应的十几行。记录输入、输出、损失和检查点路径。熟悉后，再尝试只修改一个设计变量，例如头数、宽度或可见模态。一次改变太多内容，会使结果难以归因。

## A.2 两个不会更新参数的数学检查

```bash
python code/verify_examples.py
python code/mini_transformer.py
```

这两个命令只需要NumPy。它们验证注意力归一化、因果遮蔽、GPT-2参数总数、KV容量、MLA乘法重排、RoPE相对角度及微型解码器缓存。NumPy解码器使用随机参数，没有训练功能；它的用途是检验“这个计算是否实现了我以为的公式”。这与后面的PyTorch训练应分开理解。

缓存检查比较三种方式：一次处理完整序列；逐token生成并保存每层KV；先做多token prefill，再用缓存处理下一段。三种结果应在浮点误差范围内一致。还会修改最后一个输入token，检查之前位置的logits是否不变。这类干预比只打印shape更能发现因果掩码错误。

## A.3 每章训练的验收标准

第01-11章统一运行`python -m tinyomni.lab --chapter 章节号`，第12-16章切换到`python -m tinyomni.train`。前11章保留独立模型，是为了让某个机制的成败可以单独检查，并不是把所有小模型的权重直接拼成TinyOmni。最终项目在一个共同模型中重新训练。

运行小实验后，至少检查四件事：训练loss是否有限；参数是否真的变化；独立数据上的指标是否改善；保存后重新加载的输出是否一致。`lab.py`会对参数变化和重载等价性做断言，并保存`model.pt`、`metrics.json`、`loss.csv`。TinyOmni则额外保存优化器状态，支持恢复训练。

遇到很高的准确率，不要立即提高难度并宣布成功。第05章周期序列只有少量语义模式，第11章规则教师只有100种数字对。它们适合确认实现，却不能证明开放语言建模或通用推理。训练一个机制实验与训练一个具备广泛能力的模型，验证问题不同。

## A.4 显式输入，而不是只抽取一个seed

标准推理CLI按seed抽取可核对样本，便于比较目标与预测。还可以使用`custom_infer`直接指定受控文字条件：

```bash
python -m tinyomni.custom_infer --checkpoint runs/tinyomni-final/checkpoint.pt --text "blue square left" --device cpu --out runs/custom-text
```

把`--device cpu`改为`cuda`即可在可用的GPU上运行。文字语法严格为“颜色 形状 方向”，可用词见第17章。这里仍是17个词表条目的结构化语言，不是任意中文或英文聊天。

也可把标准demo导出的输入文件重新读入，验证文件部署链：

```bash
python -m tinyomni.custom_infer --checkpoint runs/tinyomni-final/checkpoint.pt --image runs/tinyomni-final/demo-image/input_image.png --device cpu --out runs/custom-image
python -m tinyomni.custom_infer --checkpoint runs/tinyomni-final/checkpoint.pt --audio runs/tinyomni-final/demo-audio/input_audio.wav --device cpu --out runs/custom-audio
```

图片必须是16×16，波形必须为单声道、8000 Hz、16位PCM、恰好256个样本。程序拒绝悄悄把任意照片缩小或把语音截取成短音调，因为这样的输入变换没有解决分布不匹配。自定义输入没有ground truth，所以它只报告预测，不伪造准确率。

## A.5 怎样设计你自己的下一个实验

先提出一个会失败的假设，再设计数据。例如，假设模型只是根据颜色猜形状，那么令颜色与形状独立均衡抽样，并检查每个组合；假设模型不看声音，那么保留文字任务标记、屏蔽音频，比较同一组样本；假设视频方向来自单帧位置，那么交换帧顺序并保持画面集合不变，看预测是否翻转。

对于生成任务，要把数值误差与可感知结构一起看。16×16图像的大部分是背景，单纯输出黑底就可能得到看似不大的MSE；音频的点对点误差受相位影响，必须先明确定义对齐与规范目标。增加现实数据时，应同步改变评测，而不是只增加训练目录里的文件数量。

最后，给每次实验留一张记录卡：问题、训练数据版本、验证划分、模型配置、随机种子、运行命令、最好检查点的选择规则、失败样本和结论。若你无法从这张卡复现结果，就还没有完成一个可交流的实验。
