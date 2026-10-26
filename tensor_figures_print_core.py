"""Numbered paper storyboards for scalar accumulation and attention.

The shared renderer injects Figure. Every stage is visible at once; grayscale
fills, explicit indices, and numbered steps carry the meaning without color.
"""

BLACK = "#111111"
GRAY = "#666666"
LIGHT = "#F0F0F0"
WHITE = "#FFFFFF"


def _grid(f, x, y, values, width=42, height=42, marks=(), size=18):
    for r, row in enumerate(values):
        for c, value in enumerate(row):
            marked = (r, c) in marks
            f.rect(x + c * width, y + r * height, width - 2, height - 2,
                   BLACK if marked else WHITE, stroke=BLACK, radius=0)
            f.txt(x + (c + .5) * width - 1,
                  y + r * height + height * .65,
                  str(value), size, WHITE if marked else BLACK, "middle")


def _step(f, n, x, y, title=None):
    f.rect(x, y, 32, 32, BLACK, stroke=BLACK, radius=0)
    f.txt(x + 16, y + 23, str(n), 20, WHITE, "middle")
    if title:
        f.txt(x + 44, y + 23, title, 18, BLACK)


def _panel(f, x, y, width, height):
    f.rect(x, y, width, height, WHITE, stroke=GRAY, radius=0)


def _figure(Figure, title, subtitle, height):
    f = Figure(title, height=height)
    f.txt(24, 57, subtitle, 16, BLACK)
    return f


def build(Figure):
    out = {}

    f = _figure(Figure, "固定一个输出格，连续累加四次",
                "读图方向：从上到下。四步都在更新同一个 Y[0,0]。", 700)
    f.txt(28, 125, "X 第0行", 18, BLACK)
    f.txt(28, 187, "W 第0列", 18, BLACK)
    for i in range(4):
        f.txt(169 + i * 62, 88, "k=" + str(i), 16, BLACK, "middle")
    _grid(f, 139, 101, [[1, 0, 2, -1]], width=62, height=44)
    _grid(f, 139, 163, [[1, 0, 1, -1]], width=62, height=44)
    f.txt(417, 120, "把 W 的一列横放，", 17, BLACK)
    f.txt(417, 147, "方便与 X 的一行配对。", 17, BLACK)
    f.txt(417, 185, "初始累计值：0", 18, BLACK)
    f.txt(177, 244, "X[0,k]", 16, BLACK, "middle")
    f.txt(265, 244, "W[k,0]", 16, BLACK, "middle")
    f.txt(353, 244, "本次乘积", 16, BLACK, "middle")
    f.txt(567, 244, "Y[0,0] 暂存", 16, BLACK, "middle")
    xs, ws, products, totals = [1, 0, 2, -1], [1, 0, 1, -1], [1, 0, 2, 1], [1, 1, 3, 4]
    previous = 0
    for i in range(4):
        y = 260 + i * 88
        _panel(f, 24, y - 5, 652, 76)
        _step(f, i + 1, 34, y + 13)
        f.txt(80, y + 39, "k=" + str(i), 17, BLACK)
        _grid(f, 154, y + 12, [[xs[i]]], width=48, height=44)
        f.txt(221, y + 42, "×", 20, BLACK, "middle")
        _grid(f, 242, y + 12, [[ws[i]]], width=48, height=44)
        f.txt(309, y + 42, "=", 20, BLACK, "middle")
        _grid(f, 330, y + 12, [[products[i]]], width=48, height=44)
        f.txt(441, y + 40, str(previous) + " + " + str(products[i]), 18, BLACK, "middle")
        f.arrow(486, y + 34, 523, y + 34, BLACK)
        _grid(f, 535, y + 10, [[totals[i]]], width=66, height=48, marks={(0, 0)})
        previous = totals[i]
    f.txt(24, 641, "第4步的 4 才是最终输出；前面的 1、1、3 是中间累计值。", 17, BLACK)
    f.txt(24, 674, "共同轴 k 被求和消掉，固定的输出地址 [0,0] 始终不变。", 16, BLACK)
    out["01-print-matmul"] = f.d

    f = _figure(Figure, "拆头再合头：十二个数一个不少",
                "B=1、T=3、H=2、每头宽度=2；黑底格始终是原值4。", 760)
    original = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    _panel(f, 24, 82, 312, 282)
    _panel(f, 364, 82, 312, 282)
    _step(f, 1, 35, 93, "原表 X")
    _step(f, 2, 375, 93, "reshape：拆开通道")
    f.txt(79, 149, "[1,3,4] = [B,T,D]", 17, BLACK)
    f.txt(389, 149, "[1,3,2,2] = [B,T,H,d_h]", 17, BLACK)
    for c in range(4):
        f.txt(100 + c * 42, 179, "d" + str(c), 16, BLACK, "middle")
    _grid(f, 80, 192, original, marks={(1, 0)})
    for r in range(3):
        f.txt(68, 219 + r * 42, "t" + str(r), 16, BLACK, "end")
    f.txt(45, 341, "值4的地址：X[0,1,0]", 16, BLACK)
    f.txt(466, 175, "h=0", 16, BLACK, "middle")
    f.txt(550, 175, "h=1", 16, BLACK, "middle")
    _grid(f, 424, 192, original, marks={(1, 0)})
    f.line(507, 188, 507, 318, BLACK, width=3)
    for r in range(3):
        f.txt(412, 219 + r * 42, "t" + str(r), 16, BLACK, "end")
    f.txt(385, 341, "值4的地址：R[0,1,0,0]", 16, BLACK)
    f.arrow(340, 232, 359, 232, BLACK)
    f.line(520, 369, 520, 386, BLACK, width=1.6)
    f.line(520, 386, 180, 386, BLACK, width=1.6)
    f.arrow(180, 386, 180, 401, BLACK)

    _panel(f, 24, 407, 312, 300)
    _panel(f, 364, 407, 312, 300)
    _step(f, 3, 35, 418, "transpose：交换 T、H")
    _step(f, 4, 375, 418, "交换回来，再合并")
    f.txt(44, 470, "[1,2,3,2] = [B,H,T,d_h]", 17, BLACK)
    f.txt(420, 470, "[1,3,4] = [B,T,D]", 17, BLACK)
    for x, h, values, marked in [(72, 0, [[0, 1], [4, 5], [8, 9]], {(1, 0)}),
                                 (215, 1, [[2, 3], [6, 7], [10, 11]], set())]:
        f.txt(x + 37, 495, "h=" + str(h), 16, BLACK, "middle")
        _grid(f, x, 507, values, width=38, height=38, marks=marked)
        for r in range(3):
            f.txt(x - 9, 532 + r * 38, "t" + str(r), 16, BLACK, "end")
    f.txt(45, 650, "值4的地址：U[0,0,1,0]", 16, BLACK)
    f.txt(45, 682, "每个头仍保留全部3个位置。", 16, BLACK)
    _grid(f, 424, 507, original, width=38, height=38, marks={(1, 0)})
    for r in range(3):
        f.txt(412, 532 + r * 38, "t" + str(r), 16, BLACK, "end")
    for c in range(4):
        f.txt(442 + c * 38, 495, "d" + str(c), 16, BLACK, "middle")
    f.txt(385, 650, "值4的地址：X'[0,1,0]", 16, BLACK)
    f.txt(385, 682, "12个数回到原来的地址。", 16, BLACK)
    f.arrow(340, 561, 359, 561, BLACK)
    f.txt(24, 738, "黑底白字标记同一个元素；分组与换轴不会进行乘加。", 16, BLACK)
    out["03-print-heads"] = f.d

    f = _figure(Figure, "固定查询2，把注意力算完五步",
                "全程只追踪索引2的查询；显示值舍入，最终结果用完整精度。", 760)
    _panel(f, 24, 78, 652, 142)
    _step(f, 1, 34, 89, "点积：q2 与三个键比较")
    f.txt(100, 140, "q2", 16, BLACK)
    f.txt(286, 124, "K.T", 16, BLACK)
    f.txt(502, 140, "原始分数", 16, BLACK)
    _grid(f, 100, 151, [[1, 1]], width=40, height=40, marks={(0, 0), (0, 1)})
    _grid(f, 286, 133, [[1, 0, 1], [0, 1, 1]], width=40, height=40,
          marks={(0, 2), (1, 2)})
    _grid(f, 502, 151, [[1, 1, 2]], width=48, height=40, marks={(0, 2)})
    f.txt(235, 178, "@", 22, BLACK, "middle")
    f.txt(455, 178, "=", 22, BLACK, "middle")
    f.txt(470, 209, "末格：1×1 + 1×1 = 2", 16, BLACK)

    _panel(f, 24, 232, 652, 74)
    _step(f, 2, 34, 246, "每格除以 √2")
    f.txt(78, 291, "键的通道数 d_k=2", 16, BLACK)
    _grid(f, 322, 250, [[".707107", ".707107", "1.414214"]],
          width=110, height=42, size=17)

    _panel(f, 24, 318, 652, 80)
    _step(f, 3, 34, 333, "因果掩码：查询2")
    f.txt(78, 382, "没有未来格，三项均保留。", 16, BLACK)
    for c in range(3):
        f.txt(376 + c * 110, 338, "键" + str(c) + " 可读", 16, BLACK, "middle")
    _grid(f, 322, 348, [[".707107", ".707107", "1.414214"]],
          width=110, height=36, size=17)

    _panel(f, 24, 410, 652, 92)
    _step(f, 4, 34, 427, "沿键轴做 softmax")
    f.txt(78, 485, "未舍入时三项之和为1。", 16, BLACK)
    _grid(f, 322, 430, [[".248255", ".248255", ".503490"]],
          width=110, height=42, size=17)

    _panel(f, 24, 514, 652, 222)
    _step(f, 5, 34, 525, "缩放 V 的三行，再按列相加")
    f.txt(100, 567, "权重", 16, BLACK)
    f.txt(240, 567, "值向量", 16, BLACK)
    f.txt(404, 567, "各行的贡献", 16, BLACK)
    _grid(f, 100, 576, [[".248255"], [".248255"], [".503490"]],
          width=100, height=32, size=17)
    _grid(f, 240, 576, [[1, 0], [0, 2], [3, 1]], width=48, height=32, size=17)
    _grid(f, 404, 576, [[".248255", "0"], ["0", ".496510"], ["1.510470", ".503490"]],
          width=112, height=32, size=17)
    for r in range(3):
        f.txt(217, 599 + r * 32, "×", 18, BLACK, "middle")
        f.txt(366, 599 + r * 32, "=", 18, BLACK, "middle")
    f.txt(78, 714, "按列求和 → O[2]", 18, BLACK)
    _grid(f, 376, 690, [["1.7587246", "1.0000000"]],
          width=130, height=34, marks={(0, 0), (0, 1)}, size=17)
    out["03-print-attention"] = f.d
    return out
