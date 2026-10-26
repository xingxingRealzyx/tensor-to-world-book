"""Complete two-batch, two-head attention storyboards for print.

All numerical slices are retained. The example uses projected Q/K/V tensors,
independent of the arange layout example. No rendering module is imported.
"""

import math

BLACK = "#111111"
WHITE = "#FFFFFF"
GRAY = "#666666"
LIGHT = "#EEEEEE"

Q = [
    [[[1, 0], [0, 1], [1, 1]], [[0, 1], [1, 0], [1, -1]]],
    [[[1, 1], [1, 0], [0, 1]], [[1, -1], [0, 1], [1, 1]]],
]
K = [
    [[[1, 0], [0, 1], [1, 1]], [[1, 1], [0, 1], [1, 0]]],
    [[[1, 0], [1, 1], [0, 1]], [[0, 1], [1, 0], [1, 1]]],
]
V = [
    [[[1, 0], [0, 2], [3, 1]], [[2, 1], [1, 3], [0, 2]]],
    [[[1, 2], [3, 0], [5, 4]], [[0, 1], [2, 3], [4, 5]]],
]
KEY_VALID = [[True, True, True], [True, True, False]]


def case_data():
    """Reference data computed with scalar loops, independently of PyTorch."""
    scores, weights, outputs, allowed = [], [], [], []
    for b in range(2):
        sb, ab, ob, mb = [], [], [], []
        for h in range(2):
            s = [[sum(Q[b][h][i][d] * K[b][h][j][d] for d in range(2))
                  for j in range(3)] for i in range(3)]
            mask = [[j <= i and KEY_VALID[b][j] for j in range(3)] for i in range(3)]
            a = []
            for i in range(3):
                high = max(s[i][j] / math.sqrt(2) for j in range(3) if mask[i][j])
                exps = [math.exp(s[i][j] / math.sqrt(2) - high) if mask[i][j] else 0.
                        for j in range(3)]
                a.append([v / sum(exps) for v in exps])
            o = [[sum(a[i][j] * V[b][h][j][d] for j in range(3))
                  for d in range(2)] for i in range(3)]
            sb.append(s); ab.append(a); ob.append(o); mb.append(mask)
        scores.append(sb); weights.append(ab); outputs.append(ob); allowed.append(mb)
    merged = [[outputs[b][0][i] + outputs[b][1][i] for i in range(3)] for b in range(2)]
    return {"Q": Q, "K": K, "V": V, "key_valid": KEY_VALID,
            "raw_scores": scores, "allowed": allowed, "A": weights,
            "O": outputs, "merged": merged}


def _grid(f, x, y, values, width=40, height=None, marks=(), blocked=(), size=16):
    height = height or width
    for r, row in enumerate(values):
        for c, value in enumerate(row):
            marked = (r, c) in marks
            f.rect(x + c * width, y + r * height, width - 2, height - 2,
                   BLACK if marked else LIGHT if (r, c) in blocked else WHITE,
                   stroke=BLACK, radius=0)
            f.txt(x + (c + .5) * width - 1, y + (r + .65) * height,
                  str(value), size, WHITE if marked else BLACK, "middle")


def _fmt(values, places=3):
    return [[str(int(v)) if float(v).is_integer() else f"{v:.{places}f}".replace("0.", ".", 1)
             if 0 < v < 1 else f"{v:.{places}f}" for v in row] for row in values]


def _panel(f, x, y, width, height):
    f.rect(x, y, width, height, WHITE, stroke=GRAY, radius=0)


def _step(f, n, x, y, label):
    f.rect(x, y, 30, 30, BLACK, stroke=BLACK, radius=0)
    f.txt(x + 15, y + 22, str(n), 19, WHITE, "middle")
    f.txt(x + 42, y + 22, label, 18, BLACK)


def _figure(Figure, title, subtitle, height=760):
    f = Figure(title, height=height)
    f.txt(24, 57, subtitle, 16, BLACK)
    return f


def build(Figure):
    out = {}
    data = case_data()
    f = _figure(Figure, "两份样本各拆两头：完整保留24格",
                "布局编号例：X=arange(24).reshape(2,3,4)；黑底追踪14和18。")
    originals = [[[12*b + 4*t+d for d in range(4)] for t in range(3)] for b in range(2)]
    _step(f, 1, 24, 82, "X")
    f.txt(24, 141, "[2,3,4]", 18, BLACK)
    f.txt(24, 174, "[B,T,D]", 17, BLACK)
    for b, x in enumerate([200, 455]):
        f.txt(x, 97, "样本 b=" + str(b), 17, BLACK)
        _grid(f, x, 109, originals[b], width=32,
              marks={(0, 2), (1, 2)} if b else ())
    f.arrow(80, 211, 80, 229, BLACK)
    _step(f, 2, 24, 239, "R")
    f.txt(24, 298, "[2,3,2,2]", 18, BLACK)
    f.txt(24, 331, "[B,T,H,d_h]", 17, BLACK)
    for b, x in enumerate([200, 455]):
        f.txt(x, 234, "样本 b=" + str(b), 17, BLACK)
        f.txt(x + 31, 253, "h=0", 16, BLACK, "middle")
        f.txt(x + 95, 253, "h=1", 16, BLACK, "middle")
        _grid(f, x, 265, originals[b], width=32,
              marks={(0, 2), (1, 2)} if b else ())
        f.line(x + 63, 260, x + 63, 361, BLACK, width=3)
    f.arrow(80, 369, 80, 388, BLACK)
    _step(f, 3, 24, 399, "Q_layout")
    f.txt(24, 458, "[2,2,3,2]", 18, BLACK)
    f.txt(24, 491, "[B,H,T,d_h]", 17, BLACK)
    f.txt(291, 428, "h=0", 18, BLACK, "middle")
    f.txt(523, 428, "h=1", 18, BLACK, "middle")
    for b, y in enumerate([444, 568]):
        f.txt(173, y + 55, "b=" + str(b), 18, BLACK)
        for h, x in enumerate([260, 492]):
            values = [[12*b + 4*t + 2*h + d for d in range(2)] for t in range(3)]
            _grid(f, x, y, values, width=32,
                  marks={(0, 0), (1, 0)} if (b, h) == (1, 1) else ())
            for t in range(3):
                f.txt(x - 9, y + (t+.65)*32, "t"+str(t), 16, BLACK, "end")
    f.txt(24, 702, "14：[1,0,2] → [1,0,1,0] → [1,1,0,0]", 17, BLACK)
    f.txt(24, 735, "18：[1,1,2] → [1,1,1,0] → [1,1,1,0]", 17, BLACK)
    out["03-4d-layout"] = f.d

    f = _figure(Figure, "四维矩阵乘法：四片各算各的",
                "Q[2,2,3,2] @ K.transpose(-2,-1)[2,2,2,3] → S_raw[2,2,3,3]", 730)
    for b, y in enumerate([100, 390]):
        for h, x in enumerate([24, 366]):
            _panel(f, x, y, 310, 270)
            f.txt(x+16, y+28, f"固定 b={b}, h={h}", 18, BLACK)
            for xx, name in [(12, "Q"), (104, "K.T"), (218, "S_raw")]:
                f.txt(x+xx, y+69, name, 16, BLACK)
            marked = (b, h) == (1, 0)
            kt = [list(row) for row in zip(*K[b][h])]
            _grid(f, x+12, y+90, Q[b][h], width=30,
                  marks={(2, 0), (2, 1)} if marked else ())
            _grid(f, x+104, y+105, kt, width=30,
                  marks={(0, 1), (1, 1)} if marked else ())
            _grid(f, x+218, y+90, data["raw_scores"][b][h], width=30,
                  marks={(2, 1)} if marked else ())
            f.txt(x+86, y+145, "@", 18, BLACK, "middle")
            f.txt(x+206, y+145, "=", 18, BLACK, "middle")
            if marked:
                f.txt(x+16, y+221, "S_raw[1,0,2,1]", 17, BLACK)
                f.txt(x+16, y+248, "= 0×1 + 1×1 = 1", 17, BLACK)
            else:
                f.txt(x+16, y+221, "S 行：查询位置 i", 17, BLACK)
                f.txt(x+16, y+248, "S 列：键位置 j", 17, BLACK)
    f.txt(24, 699, "只转置 K 的最后两轴；B、H 留作配对索引，不参与求和。", 16, BLACK)
    out["03-4d-scores"] = f.d

    f = _figure(Figure, "两种掩码广播到四片分数表",
                "1=允许，0=禁止；合并规则为逻辑与，之后沿最后键轴做softmax。")
    f.txt(24, 88, "因果 [1,1,3,3]", 17, BLACK)
    f.txt(380, 88, "键有效 [2,1,1,3]", 17, BLACK)
    _grid(f, 48, 113, [[1, 0, 0], [1, 1, 0], [1, 1, 1]], width=38)
    _grid(f, 420, 113, [[1, 1, 1], [1, 1, 0]], width=54, height=40)
    f.txt(408, 139, "b0", 16, BLACK, "end")
    f.txt(408, 179, "b1", 16, BLACK, "end")
    f.txt(380, 220, "b1 的键2是 padding。", 16, BLACK)
    f.txt(24, 260, "允许 = 因果 AND 键有效；禁止格先设为 −inf。", 17, BLACK)
    f.txt(24, 290, "softmax(..., dim=-1) → A[2,2,3,3]；0× 表示被掩码置零。", 16, BLACK)
    f.txt(224, 317, "h=0", 18, BLACK, "middle")
    f.txt(529, 317, "h=1", 18, BLACK, "middle")
    for b, y in enumerate([330, 519]):
        f.txt(35, y+83, "b="+str(b), 18, BLACK)
        for h, x in enumerate([150, 455]):
            values = _fmt(data["A"][b][h])
            blocked = []
            for i in range(3):
                for j in range(3):
                    if not data["allowed"][b][h][i][j]:
                        values[i][j] = "0×"; blocked.append((i, j))
            _grid(f, x, y, values, width=50, marks={(2,0),(2,1)} if (b,h)==(1,0) else (), blocked=blocked)
            for i in range(3):
                f.txt(x-9, y+(i+.65)*50, "q"+str(i), 16, BLACK, "end")
    f.txt(24, 708, "样本1的键2在所有头、所有查询中被屏蔽；查询2行仍有权重。", 16, BLACK)
    f.txt(24, 739, "本例每行至少有键0可读，不存在全屏蔽行。显示权重保留三位。", 16, BLACK)
    out["03-4d-masks"] = f.d

    f = _figure(Figure, "四片权重各读自己的值表",
                "A[2,2,3,3] @ V[2,2,3,2] → O[2,2,3,2]；沿键轴求和。")
    f.txt(136, 93, "A：查询×键", 16, BLACK)
    f.txt(338, 93, "V：键×值", 16, BLACK)
    f.txt(500, 93, "O：查询×值", 16, BLACK)
    for index, (b,h) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
        y = 106 + index*146
        _panel(f, 24, y-6, 652, 144)
        f.txt(40, y+49, "b="+str(b), 18, BLACK)
        f.txt(40, y+80, "h="+str(h), 18, BLACK)
        marked = (b,h)==(1,0)
        _grid(f, 136, y, _fmt(data["A"][b][h]), width=44,
              marks={(2,0),(2,1),(2,2)} if marked else ())
        _grid(f, 338, y, V[b][h], width=44,
              marks={(0,0),(1,0),(2,0)} if marked else ())
        _grid(f, 500, y, _fmt(data["O"][b][h]), width=60, height=44,
              marks={(2,0)} if marked else ())
        f.txt(302, y+78, "@", 22, BLACK, "middle")
        f.txt(462, y+78, "=", 22, BLACK, "middle")
    f.txt(24, 716, "O[1,0,2,0] = .330238…×1 + .669761…×3 + 0×5 ≈ 2.339523", 16, BLACK)
    f.txt(24, 741, "各表显示三位小数；结果由完整精度计算，不按显示值重算。", 16, BLACK)
    out["03-4d-read"] = f.d

    f = _figure(Figure, "合头：同一样本同一位置并排接回",
                "先 O.transpose(1,2)，再把 H 与 d_v 合并；没有跨样本的加法。")
    f.txt(240, 92, "h=0", 18, BLACK, "middle")
    f.txt(530, 92, "h=1", 18, BLACK, "middle")
    for b, y in enumerate([109, 283]):
        f.txt(35, y+69, "b="+str(b), 18, BLACK)
        for h, x in enumerate([180,470]):
            _grid(f, x, y, _fmt(data["O"][b][h]), width=60, height=40,
                  marks={(2,0)} if (b,h)==(1,0) else ())
            for i in range(3):
                f.txt(x-9, y+(i+.65)*40, "q"+str(i), 16, BLACK, "end")
    f.txt(24, 446, "[2,2,3,2] → [2,3,2,2] → [2,3,4]", 20, BLACK)
    f.txt(58, 490, "合回：样本 b=0", 18, BLACK)
    f.txt(389, 490, "合回：样本 b=1", 18, BLACK)
    for b, x in enumerate([58,389]):
        _grid(f, x, 508, _fmt(data["merged"][b]), width=64, height=42,
              marks={(2,0)} if b else ())
        f.line(x+127, 504, x+127, 636, BLACK, width=3)
        f.txt(x+63, 660, "头0的两维", 16, BLACK, "middle")
        f.txt(x+191, 660, "头1的两维", 16, BLACK, "middle")
    f.txt(24, 701, "O[1,0,2,0] ≈ 2.339523 → merged[1,2,0]，数值不变。", 17, BLACK)
    f.txt(24, 735, "粗线只标出原来的头边界；后续 W_O 才进行通道混合。", 16, BLACK)
    out["03-4d-merge"] = f.d
    return out
