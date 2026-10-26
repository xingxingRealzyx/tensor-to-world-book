"""Cell-by-cell tensor illustrations for Chapters 05--11.

The shared renderer injects Figure. This module deliberately does not import
build_figures, so it can also be rendered and checked independently.
"""

BLUE = "#235B93"
TEAL = "#087F82"
ORANGE = "#CC702D"
INK = "#172D40"
MUTED = "#536D7E"
PALE_BLUE = "#DCEAF6"
PALE_TEAL = "#D9EFF0"
PALE_ORANGE = "#FFE6C5"
GRAY = "#E2E7EC"


def _grid(f, x, y, values, cell=46, rows=None, cols=None, marks=None, size=16):
    """A readable scalar grid; selected rows/columns are real colored cells."""
    rows, cols, marks = rows or {}, cols or {}, marks or {}
    for r, row in enumerate(values):
        for c, value in enumerate(row):
            fill = marks.get((r, c), cols.get(c, rows.get(r, "#F0F4F7")))
            border = ORANGE if (r, c) in marks else "#B9CBD8"
            f.rect(x + c * cell, y + r * cell, cell - 2, cell - 2,
                   fill, stroke=border, radius=2)
            f.txt(x + (c + .5) * cell - 1, y + r * cell + cell * .63,
                  str(value), size, INK, "middle")


def _cols(f, x, y, names, cell):
    for c, name in enumerate(names):
        f.txt(x + (c + .5) * cell - 1, y - 12, name, 14, MUTED, "middle")


def _rows(f, x, y, names, cell):
    for r, name in enumerate(names):
        f.txt(x - 9, y + (r + .63) * cell, name, 14, MUTED, "end")


def _figure(Figure, title, subtitle, height):
    f = Figure(title, height=height)
    f.txt(24, 57, subtitle, 14, MUTED)
    return f


def build(Figure):
    out = {}

    f = _figure(Figure, "一行表示怎样给全部候选打分", "一张样本切片：T=3、D=3、V=3；橙格是本次追踪的结果。", 590)
    n = [[1, 1, 0], [0, 1, 1], [1, 0, 2]]
    et = [[1, 0, -1], [0, 1, 0], [0, 1, 1]]
    logits = [[1, 1, -1], [0, 2, 1], [1, 2, 1]]
    f.txt(54, 89, "隐藏表示 N", 17, BLUE)
    f.txt(285, 89, "词表转置 E.T", 17, TEAL)
    f.txt(504, 89, "logits", 17, ORANGE)
    _cols(f, 54, 122, ["d0", "d1", "d2"], 46)
    _cols(f, 285, 122, ["A", "B", "C"], 46)
    _cols(f, 504, 122, ["A", "B", "C"], 46)
    _rows(f, 54, 122, ["t0", "t1", "t2"], 46)
    _grid(f, 54, 122, n, rows={2: PALE_BLUE})
    _grid(f, 285, 122, et, cols={1: PALE_TEAL})
    _grid(f, 504, 122, logits, rows={2: PALE_ORANGE}, marks={(2, 1): "#FFD29A"})
    f.txt(233, 196, "@", 25, INK, "middle")
    f.txt(465, 196, "=", 25, INK, "middle")
    f.txt(54, 302, "展开追踪 logits[t2,B]：", 17, ORANGE)
    _grid(f, 54, 350, [[1, 0, 2]], rows={0: PALE_BLUE})
    f.txt(54, 333, "取 N 的最后有效行", 14, BLUE)
    _grid(f, 285, 324, [[0], [1], [1]], cols={0: PALE_TEAL})
    f.txt(285, 307, "取 E.T 的 B 列", 14, TEAL)
    _grid(f, 515, 367, [[2]], cell=54, marks={(0, 0): "#FFD29A"}, size=20)
    f.txt(235, 384, "@", 25, INK, "middle")
    f.arrow(344, 391, 500, 391, ORANGE)
    f.txt(350, 492, "1×0 + 0×1 + 2×1 = 2", 21, ORANGE, "middle")
    f.txt(24, 528, "生成时取最后有效行 [1,2,1]，沿候选轴做 softmax。", 16, INK)
    f.txt(24, 552, "最后一行说的是位置；最大分数说的是候选，两次选择不要混淆。", 14, MUTED)
    out["05-tensor-vocab"] = f.d

    f = _figure(Figure, "标签选中概率表里的哪一格", "先把标签错开一位；再由每行标签选一格，最后用 loss mask 计分。", 640)
    f.txt(200, 87, "完整片段", 16, BLUE)
    _grid(f, 200, 104, [["a", "b", "c", "d", "EOS"]], cell=50,
          cols={1: PALE_TEAL, 2: PALE_TEAL, 3: PALE_TEAL, 4: PALE_TEAL})
    f.arrow(219, 158, 250, 200, BLUE)
    f.arrow(421, 158, 410, 200, TEAL)
    f.txt(70, 190, "输入 x：前四格", 15, BLUE)
    f.txt(448, 190, "标签 y：后四格", 15, TEAL)
    _grid(f, 70, 208, [["a", "b", "c", "d"]], cell=48, rows={0: PALE_BLUE})
    _grid(f, 400, 208, [["b", "c", "d", "EOS"]], cell=48, rows={0: PALE_TEAL})
    p = [[".05", ".80", ".05", ".05", ".05"],
         [".05", ".10", ".50", ".15", ".20"],
         [".10", ".20", ".20", ".25", ".25"],
         [".05", ".10", ".10", ".05", ".70"]]
    f.txt(54, 285, "预测概率 P[t,候选]", 16, BLUE)
    _cols(f, 54, 315, ["a", "b", "c", "d", "EOS"], 52)
    _rows(f, 54, 315, ["t0", "t1", "t2", "t3"], 52)
    _grid(f, 54, 315, p, cell=52, rows={3: GRAY},
          marks={(0, 1): PALE_ORANGE, (1, 2): "#FFD29A", (2, 3): PALE_ORANGE}, size=15)
    f.txt(369, 299, "loss mask", 14, MUTED, "middle")
    f.txt(473, 299, "选中概率", 14, TEAL, "middle")
    f.txt(595, 299, "−ln(p)", 15, ORANGE, "middle")
    _grid(f, 342, 315, [[1], [1], [1], [0]], cell=52, rows={3: GRAY})
    _grid(f, 446, 315, [[".80"], [".50"], [".25"], [".70"]], cell=52,
          rows={0: PALE_TEAL, 1: PALE_TEAL, 2: PALE_TEAL, 3: GRAY}, size=15)
    _grid(f, 568, 315, [[".2231"], [".6931"], ["1.3863"], ["忽略"]], cell=52,
          rows={3: GRAY}, marks={(1, 0): "#FFD29A"}, size=14)
    f.arrow(505, 397, 558, 397, ORANGE)
    f.txt(350, 574, "loss ≈ (.2231 + .6931 + 1.3863) / 3 = .7675", 17, ORANGE, "middle")
    f.txt(24, 611, "本例只忽略最后一行的损失；它并不替代注意力的因果掩码。", 15, INK)
    out["06-tensor-shift-loss"] = f.d

    f = _figure(Figure, "缓存就是留下旧行再追加新行", "固定一个样本、一个层、一个头；右侧缓存已经包含本轮新位置。", 590)
    old_k = [[1, 0], [0, 1], [1, 1]]
    old_v = [[2, 0], [0, 2], [1, 3]]
    for x, label, values in [(60, "旧 K", old_k), (210, "旧 V", old_v)]:
        f.txt(x, 98, label, 17, BLUE)
        _grid(f, x, 120, values, cell=44, rows={0: PALE_BLUE, 1: PALE_BLUE, 2: PALE_BLUE})
    _rows(f, 60, 120, ["t0", "t1", "t2"], 44)
    for x, label, values in [(60, "新 k3", [[2, 1]]), (210, "新 v3", [[4, 1]])]:
        f.txt(x, 291, label, 16, ORANGE)
        _grid(f, x, 309, values, cell=44, rows={0: PALE_ORANGE})
    f.txt(414, 98, "K_cache", 17, TEAL)
    f.txt(564, 98, "V_cache", 17, TEAL)
    _grid(f, 414, 120, old_k + [[2, 1]], cell=44,
          rows={0: PALE_BLUE, 1: PALE_BLUE, 2: PALE_BLUE, 3: PALE_ORANGE})
    _grid(f, 564, 120, old_v + [[4, 1]], cell=44,
          rows={0: PALE_BLUE, 1: PALE_BLUE, 2: PALE_BLUE, 3: PALE_ORANGE})
    f.txt(356, 169, "按行拼接", 15, MUTED, "middle")
    f.arrow(311, 192, 400, 192, TEAL)
    f.arrow(302, 329, 400, 278, ORANGE)
    f.txt(414, 327, "前三行复用，只有橙色行是新的。", 14, MUTED)
    f.txt(60, 399, "当前 q=[1,0]", 16, BLUE)
    _grid(f, 60, 418, [[1, 0]], cell=44, rows={0: PALE_BLUE})
    f.txt(178, 446, "@ K_cache.T", 17, TEAL)
    f.arrow(306, 441, 341, 441, TEAL)
    _cols(f, 355, 418, ["t0", "t1", "t2", "t3"], 64)
    _grid(f, 355, 418, [[1, 0, 1, 2]], cell=64, marks={(0, 3): "#FFD29A"})
    f.txt(350, 517, "新格 s3 = 1×2 + 0×1 = 2（此处尚未缩放）", 18, ORANGE, "middle")
    f.txt(24, 552, "缓存少做旧位置的计算；当前查询仍然需要读全部四行。", 15, INK)
    out["07-tensor-cache-append"] = f.d

    f = _figure(Figure, "两个查询头可以共读一份键和值", "每头一个查询；下方 A = softmax(S/sqrt(2))，按各头自己的行归一化。", 590)
    f.txt(55, 94, "两头当前 Q", 16, BLUE)
    f.txt(265, 94, "共享 K.T", 16, TEAL)
    f.txt(500, 94, "未缩放分数 S", 16, ORANGE)
    _rows(f, 55, 120, ["h0", "h1"], 46)
    _grid(f, 55, 120, [[1, 0], [0, 1]], rows={0: PALE_BLUE, 1: PALE_TEAL})
    _cols(f, 265, 120, ["t0", "t1", "t2"], 46)
    _grid(f, 265, 120, [[1, 0, 1], [0, 1, 1]], cols={0: PALE_TEAL})
    _grid(f, 500, 120, [[1, 0, 1], [0, 1, 1]], marks={(0, 0): PALE_ORANGE})
    f.txt(202, 176, "@", 25, INK, "middle")
    f.txt(460, 176, "=", 25, INK, "middle")
    f.txt(55, 262, "Q 不同，分数和权重就可以不同；共享不等于输出相同。", 16, INK)
    f.txt(40, 316, "两头各自的 A", 16, BLUE)
    f.txt(310, 291, "共享 V", 16, TEAL)
    f.txt(520, 316, "各自输出 O", 16, ORANGE)
    _grid(f, 40, 340, [[".4011", ".1978", ".4011"], [".1978", ".4011", ".4011"]],
          cell=60, rows={0: PALE_BLUE, 1: PALE_TEAL}, size=14)
    _grid(f, 310, 315, [[2, 0], [0, 2], [1, 3]], cell=50, cols={0: PALE_TEAL})
    _grid(f, 520, 340, [["1.2033", "1.5989"], [".7967", "2.0056"]],
          cell=65, marks={(0, 0): PALE_ORANGE}, size=14)
    f.txt(265, 406, "@", 25, INK, "middle")
    f.txt(470, 406, "=", 25, INK, "middle")
    f.txt(350, 521, "O[h0,0] ≈ 2×.4011 + 0×.1978 + 1×.4011 = 1.2033", 17, ORANGE, "middle")
    f.txt(24, 557, "同一份 V 被两行权重读取；真实输出使用未舍入的权重计算。", 15, INK)
    out["07-tensor-gqa-share"] = f.d

    f = _figure(Figure, "门控是同一位置的格子逐一相乘", "只跟踪一个 token：先投影，再同格相乘，最后混合回两个通道。", 640)
    f.txt(55, 94, "输入 x", 16, BLUE)
    f.txt(263, 83, "W_gate", 16, TEAL)
    f.txt(502, 94, "z=x@W_gate", 16, BLUE)
    _grid(f, 55, 115, [[1, 2]], cell=50, rows={0: PALE_BLUE})
    _grid(f, 263, 96, [[1, 0, -1], [0, 1, 1]], cell=50, cols={2: PALE_TEAL})
    _grid(f, 502, 115, [[1, 2, 1]], cell=50, marks={(0, 2): PALE_ORANGE})
    f.txt(209, 148, "@", 25, INK, "middle")
    f.txt(458, 148, "=", 25, INK, "middle")
    f.txt(34, 253, "U=x@W_up", 16, TEAL)
    f.txt(263, 253, "G=SiLU(z)", 16, BLUE)
    f.txt(494, 253, "M=U * G", 16, ORANGE)
    _grid(f, 34, 277, [[3, 1, 2]], cell=58, cols={2: PALE_TEAL})
    _grid(f, 263, 277, [[".7311", "1.7616", ".7311"]], cell=58,
          cols={2: PALE_BLUE}, size=14)
    _grid(f, 494, 277, [["2.1932", "1.7616", "1.4621"]], cell=58,
          marks={(0, 2): PALE_ORANGE}, size=14)
    f.txt(236, 313, "*", 26, INK, "middle")
    f.txt(467, 313, "=", 25, INK, "middle")
    f.txt(350, 373, "追第三格：2 × SiLU(1) ≈ 1.4621", 19, ORANGE, "middle")
    f.txt(40, 416, "M", 16, BLUE)
    f.txt(300, 391, "W_down", 16, TEAL)
    f.txt(520, 416, "返回主干 F", 16, ORANGE)
    _grid(f, 40, 438, [["2.1932", "1.7616", "1.4621"]], cell=58, size=14)
    _grid(f, 300, 410, [[1, 0], [0, 1], [1, -1]], cell=43, cols={0: PALE_TEAL})
    _grid(f, 520, 438, [["3.6553", ".2995"]], cell=65,
          marks={(0, 0): PALE_ORANGE}, size=14)
    f.txt(256, 476, "@", 25, INK, "middle")
    f.txt(463, 476, "=", 25, INK, "middle")
    f.txt(350, 579, "F[0] ≈ 2.1932×1 + 1.7616×0 + 1.4621×1 = 3.6553", 16, ORANGE, "middle")
    f.txt(24, 606, "G 中的 1.7616 大于 1：这里的门并不是概率或二值开关。", 14, MUTED)
    out["08-tensor-swiglu"] = f.d

    f = _figure(Figure, "专家先收集若干行再加回原行", "沿用第09.2节路由；t1会被两名专家读取，但最终仍回到同一个t1。", 620)
    f.txt(56, 95, "原位置 X", 16, BLUE)
    f.txt(225, 95, "E1：取 t1,t3,t4", 14, TEAL)
    f.txt(393, 95, "E2：取 t1,t2", 14, TEAL)
    f.txt(548, 95, "E3：取 t2,t3,t4", 14, TEAL)
    _grid(f, 56, 127, [[1, 2], [2, 0], [-1, 1], [0, 2]], cell=40,
          rows={0: PALE_BLUE})
    _rows(f, 56, 127, ["t1", "t2", "t3", "t4"], 40)
    for x, values, names in [(240, [[1, 2], [-1, 1], [0, 2]], ["t1", "t3", "t4"]),
                             (400, [[1, 2], [2, 0]], ["t1", "t2"]),
                             (560, [[2, 0], [-1, 1], [0, 2]], ["t2", "t3", "t4"])]:
        _grid(f, x, 127, values, cell=40, rows={0: PALE_BLUE} if names[0] == "t1" else {})
        _rows(f, x, 127, names, 40)
    f.arrow(142, 307, 316, 307, TEAL)
    f.txt(151, 293, "gather：按索引收集行", 14, TEAL)
    f.txt(377, 292, "专家计算：E1(x)=x", 15, INK)
    f.txt(377, 315, "E2(x)=2x，E3(x)=−x", 15, INK)
    f.txt(56, 369, "t1 经 E1 的加权贡献", 15, BLUE)
    f.txt(270, 369, "t1 经 E2 的加权贡献", 15, TEAL)
    _grid(f, 56, 390, [["2/3", "4/3"]], cell=56, cols={0: PALE_BLUE})
    _grid(f, 270, 390, [["2/3", "4/3"]], cell=56, cols={0: PALE_TEAL})
    f.txt(220, 425, "+", 24, INK, "middle")
    f.arrow(389, 417, 489, 417, ORANGE)
    f.txt(503, 359, "scatter-add 回 R", 15, ORANGE)
    _grid(f, 520, 380, [["4/3", "8/3"], ["8/3", 0], ["1/2", "−1/2"], [0, "2/9"]],
          cell=44, marks={(0, 0): PALE_ORANGE}, size=15)
    _rows(f, 520, 380, ["t1", "t2", "t3", "t4"], 44)
    f.txt(56, 487, "R[t1,0] = (2/3)×1 + (1/3)×2", 17, ORANGE)
    f.txt(56, 516, "             = 4/3", 19, ORANGE)
    f.txt(24, 586, "这里只画路由输出 R；共享专家输出和残差随后再按原行相加。", 15, INK)
    out["09-tensor-gather-scatter"] = f.d

    f = _figure(Figure, "潜变量先缩成小表再按需展开", "教学例：一行从3个数降到2个；暂时省略Norm和位置支路。", 620)
    f.txt(40, 93, "输入 X [2,3]", 16, BLUE)
    f.txt(280, 93, "W_D [3,2]", 16, TEAL)
    f.txt(490, 93, "潜变量 C [2,2]", 16, ORANGE)
    _grid(f, 40, 132, [[1, 0, 1], [0, 2, 0]], cell=44, rows={1: PALE_BLUE})
    _grid(f, 280, 110, [[1, 0], [0, 1], [0, 0]], cell=44, cols={1: PALE_TEAL})
    _grid(f, 490, 132, [[1, 0], [0, 2]], cell=54, marks={(1, 1): PALE_ORANGE})
    f.txt(221, 188, "@", 25, INK, "middle")
    f.txt(435, 188, "=", 25, INK, "middle")
    f.txt(350, 278, "C[1,1] = 0×0 + 2×1 + 0×0 = 2", 19, ORANGE, "middle")
    f.txt(40, 334, "同一份 C", 16, BLUE)
    f.txt(265, 334, "U_K [2,2]", 16, TEAL)
    f.txt(490, 334, "内容键 K [2,2]", 16, ORANGE)
    _grid(f, 40, 355, [[1, 0], [0, 2]], cell=54, rows={1: PALE_BLUE})
    _grid(f, 265, 355, [[1, 0], [1, 1]], cell=54, cols={0: PALE_TEAL})
    _grid(f, 490, 355, [[1, 0], [2, 2]], cell=54, marks={(1, 0): PALE_ORANGE})
    f.txt(211, 421, "@", 25, INK, "middle")
    f.txt(434, 421, "=", 25, INK, "middle")
    f.txt(350, 506, "K[1,0] = 0×1 + 2×1 = 2", 19, ORANGE, "middle")
    f.txt(24, 551, "值表 V = C @ U_V 同理；键和值共同由小表 C 决定。", 16, INK)
    f.txt(24, 583, "小表保存什么信息由训练决定；不是任意大表都能无损压进去。", 14, MUTED)
    out["10-tensor-low-rank"] = f.d

    f = _figure(Figure, "换乘法顺序也能得到同一格", "固定前图的 C 与 U_K；两条路径比较的是softmax之前的未缩放分数。", 600)
    f.txt(40, 94, "展开路径：当前 q", 16, BLUE)
    f.txt(275, 94, "展开后的 K.T", 16, TEAL)
    f.txt(520, 94, "分数 s", 16, ORANGE)
    _grid(f, 40, 140, [[1, 2]], cell=58, rows={0: PALE_BLUE})
    _grid(f, 275, 113, [[1, 2], [0, 2]], cell=54, cols={1: PALE_TEAL})
    _grid(f, 520, 140, [[1, 6]], cell=58, marks={(0, 1): PALE_ORANGE})
    f.txt(220, 179, "@", 25, INK, "middle")
    f.txt(460, 179, "=", 25, INK, "middle")
    f.txt(350, 258, "橙格：1×2 + 2×2 = 6", 19, ORANGE, "middle")
    f.txt(40, 310, "把当前查询先变换：q @ U_K.T = [1,3]", 17, BLUE)
    f.txt(40, 355, "吸收路径：q_latent", 16, BLUE)
    f.txt(275, 355, "缓存小表 C.T", 16, TEAL)
    f.txt(520, 355, "同一行分数", 16, ORANGE)
    _grid(f, 40, 402, [[1, 3]], cell=58, rows={0: PALE_BLUE})
    _grid(f, 275, 375, [[1, 0], [0, 2]], cell=54, cols={1: PALE_TEAL})
    _grid(f, 520, 402, [[1, 6]], cell=58, marks={(0, 1): PALE_ORANGE})
    f.txt(220, 441, "@", 25, INK, "middle")
    f.txt(460, 441, "=", 25, INK, "middle")
    f.txt(350, 521, "橙格：1×0 + 3×2 = 6", 19, ORANGE, "middle")
    f.txt(24, 569, "下方不用为历史位置展开 K；位置支路须另加，不能随意移动RoPE。", 14, INK)
    out["10-tensor-absorb"] = f.d

    f = _figure(Figure, "奖励按题分行再比较同一行", "每一行是一道题，每一列是一次回答；标准差使用总体定义。", 640)
    f.txt(100, 92, "奖励矩阵 R", 17, BLUE)
    _cols(f, 100, 127, ["回答1", "回答2", "回答3", "回答4"], 52)
    _rows(f, 100, 127, ["题A", "题B"], 52)
    _grid(f, 100, 127, [[1, 0, 0, 0], [1, 1, 0, 0]], cell=52,
          rows={0: PALE_BLUE, 1: PALE_TEAL}, marks={(0, 0): PALE_ORANGE})
    f.txt(378, 108, "各行均值", 15, TEAL)
    f.txt(536, 108, "各行标准差", 15, TEAL)
    _grid(f, 378, 127, [[".25"], [".50"]], cell=52, rows={0: PALE_BLUE, 1: PALE_TEAL})
    _grid(f, 536, 127, [[".4330"], [".5000"]], cell=52,
          rows={0: PALE_BLUE, 1: PALE_TEAL}, size=14)
    f.arrow(316, 153, 366, 153, TEAL)
    f.txt(100, 281, "每行分别减均值、除以自己的标准差", 17, INK)
    f.txt(160, 303, "优势矩阵 A", 17, ORANGE)
    _rows(f, 160, 320, ["题A", "题B"], 68)
    _grid(f, 160, 320, [["1.732", "−.577", "−.577", "−.577"], [1, 1, -1, -1]],
          cell=68, rows={0: PALE_BLUE, 1: PALE_TEAL}, marks={(0, 0): PALE_ORANGE}, size=17)
    f.txt(350, 489, "A[题A,回答1] = (1−.25)/.433013 ≈ 1.732051", 18, ORANGE, "middle")
    f.txt(32, 546, "该回答的3个有效token：", 15, BLUE)
    _grid(f, 305, 511, [["1.732", "1.732", "1.732"]], cell=62,
          rows={0: PALE_ORANGE}, size=15)
    f.arrow(261, 542, 294, 542, ORANGE)
    f.txt(24, 610, "优势可正可负，不是概率；广播这份结果奖励也不等于逐步验证。", 14, INK)
    out["11-tensor-group-rewards"] = f.d

    return out
