"""Paper-first, fully populated four-axis tensor examples for chapter 01.

All diagrams retain B=2 and C=3 wherever those axes have not explicitly been
selected or reduced. Indices, outlines, and black markers carry the meaning
without color. The shared builder supplies Figure; no renderer is run here.
"""

BLACK = "#111111"
WHITE = "#FFFFFF"
LIGHT = "#EEEEEE"


def _value(b, c, h, w):
    return 24 * b + 8 * c + 4 * h + w


def _grid(f, x, y, values, cell_w=40, cell_h=40, marks=(), size=18):
    for r, row in enumerate(values):
        for c, value in enumerate(row):
            marked = (r, c) in marks
            f.rect(x + c * cell_w, y + r * cell_h, cell_w, cell_h,
                   BLACK if marked else WHITE, stroke=BLACK, radius=0)
            f.txt(x + (c + .5) * cell_w,
                  y + r * cell_h + cell_h * .66,
                  str(value), size, WHITE if marked else BLACK, "middle")


def _figure(Figure, title, subtitle, height):
    f = Figure(title, height=height)
    f.txt(24, 58, subtitle, 16, BLACK)
    return f


def _slice_panel(f, x, y, title, values, marks=(), size=18):
    """One C slice, with both spatial axes and every scalar explicitly drawn."""
    f.rect(x, y, 208, 166, LIGHT, stroke=BLACK, radius=0)
    f.txt(x + 104, y + 26, title, 17, BLACK, "middle")
    for w in range(4):
        f.txt(x + 54 + 40 * w, y + 53, "w" + str(w), 16, BLACK, "middle")
    for h in range(2):
        f.txt(x + 28, y + 91 + 40 * h, "h" + str(h), 16, BLACK, "end")
    _grid(f, x + 34, y + 64, values, marks=marks, size=size)


def build(Figure):
    out = {}

    f = _figure(Figure, "四根轴：两个样本 × 三通道 × 两行 × 四列",
                "X = arange(48).reshape(2,3,2,4)；每个小格只代表一个数。", 590)
    for b in range(2):
        y = 112 + b * 214
        f.txt(24, y - 14, "样本轴 b=" + str(b) + "：下面三片属于同一个样本", 18, BLACK)
        for c in range(3):
            values = [[_value(b, c, h, w) for w in range(4)] for h in range(2)]
            _slice_panel(f, 24 + c * 222, y, "通道轴 c=" + str(c), values,
                         marks={(1, 3)} if (b, c) == (1, 2) else ())
    f.txt(24, 531, "四维形状 [B,C,H_img,W_img] = [2,3,2,4]；共48个数。", 17, BLACK)
    f.txt(24, 565, "黑底格：先选 b=1，再选 c=2、h=1、w=3，得到47。", 17, BLACK)
    out["01-4d-slices"] = f.d

    f = _figure(Figure, "索引是逐层选择；切片可以保留轴",
                "上表每个框代表一整张2×4切片；下方展开选中的40…47。", 750)
    for c in range(3):
        f.txt(215 + c * 164, 96, "c=" + str(c), 17, BLACK, "middle")
    for b in range(2):
        f.txt(78, 138 + b * 55, "b=" + str(b), 18, BLACK, "middle")
        for c in range(3):
            x, y = 140 + c * 164, 110 + b * 55
            selected = (b, c) == (1, 2)
            f.rect(x, y, 150, 43, BLACK if selected else WHITE, stroke=BLACK, radius=0)
            f.txt(x + 75, y + 28, str(_value(b, c, 0, 0)) + " … " + str(_value(b, c, 1, 3)),
                  19, WHITE if selected else BLACK, "middle")
    f.txt(24, 241, "① 只选样本：整数索引去掉B，长度1的切片保留B。", 17, BLACK)
    for x in (24, 360):
        f.rect(x, 263, 316, 112, WHITE, stroke=BLACK, radius=0)
    f.txt(40, 292, "X[1] → [3,2,4]", 19, BLACK)
    f.txt(40, 325, "剩余轴：C、H_img、W_img", 16, BLACK)
    f.txt(40, 355, "仍有24个数；秩变为3。", 16, BLACK)
    f.txt(376, 292, "X[1:2] → [1,3,2,4]", 18, BLACK)
    f.txt(376, 325, "剩余轴：B、C、H_img、W_img", 16, BLACK)
    f.txt(376, 355, "仍有24个数；秩仍为4。", 16, BLACK)
    f.txt(24, 415, "② 再选 c=2：X[1,2]", 18, BLACK)
    for w in range(4):
        f.txt(82 + 44 * w, 447, "w" + str(w), 16, BLACK, "middle")
    _grid(f, 60, 462, [[40, 41, 42, 43], [44, 45, 46, 47]],
          cell_w=44, cell_h=44, marks={(1, 3)})
    for h in range(2):
        f.txt(50, 491 + h * 44, "h" + str(h), 16, BLACK, "end")
    f.txt(60, 579, "形状 [2,4]", 17, BLACK)
    f.arrow(257, 504, 375, 504, BLACK)
    f.txt(284, 481, "选 h=1", 16, BLACK)
    f.txt(400, 415, "③ X[1,2,1]", 18, BLACK)
    for w in range(4):
        f.txt(425 + 50 * w, 447, "w" + str(w), 16, BLACK, "middle")
    _grid(f, 400, 462, [[44, 45, 46, 47]], cell_w=50, cell_h=44, marks={(0, 3)})
    f.txt(400, 537, "形状 [4]；再选 w=3", 17, BLACK)
    f.arrow(575, 550, 575, 581, BLACK)
    _grid(f, 547, 592, [[47]], cell_w=56, cell_h=48, marks={(0, 0)}, size=22)
    f.txt(24, 625, "④ X[1,2,1,3] → 标量，形状 []，秩为0。", 17, BLACK)
    f.txt(24, 683, "若每步都用切片：X[1:2,2:3,1:2,3:4]", 17, BLACK)
    f.txt(24, 715, "结果形状 [1,1,1,1]；虽只有一个数47，四根轴仍在。", 17, BLACK)
    out["01-4d-index"] = f.d

    f = _figure(Figure, "换轴后，逐像素读出三个通道",
                "Y = X.permute(0,2,3,1)：[B,C,H_img,W_img] → [B,H_img,W_img,C]", 750)
    f.txt(24, 89, "输出形状 [2,2,4,3]；每组小格按 c0、c1、c2 排列。", 17, BLACK)
    for b in range(2):
        y = 178 + b * 176
        f.txt(24, y - 62, "b=" + str(b), 19, BLACK)
        for w in range(4):
            x = 85 + w * 148
            f.txt(x + 45, y - 36, "w=" + str(w), 17, BLACK, "middle")
            for c in range(3):
                f.txt(x + (c + .5) * 30, y - 12, "c" + str(c), 16, BLACK, "middle")
            vals = [[_value(b, c, h, w) for c in range(3)] for h in range(2)]
            marks = {(h, c) for h in range(2) for c in range(3)
                     if _value(b, c, h, w) in (41, 47)}
            _grid(f, x, y, vals, cell_w=30, cell_h=40, marks=marks, size=17)
        for h in range(2):
            f.txt(69, y + 27 + h * 40, "h=" + str(h), 17, BLACK, "end")
    f.txt(24, 480, "同形状的错误：R = X.reshape(2,2,4,3)", 18, BLACK)
    columns = [(24, "检查的输出地址"), (320, "Y：正确换轴"), (506, "R：直接重分组")]
    for x, label in columns:
        f.txt(x, 514, label, 17, BLACK)
    for y, address, correct, wrong in [(550, "[1,0,1,2]", "41", "29"),
                                       (586, "[1,1,3,2]", "47", "47")]:
        f.txt(24, y, address, 18, BLACK)
        f.txt(371, y, correct, 20, BLACK, "middle")
        f.txt(570, y, wrong, 20, BLACK, "middle")
    f.line(24, 600, 676, 600, BLACK)
    f.txt(24, 631, "固定像素 [b=1,h=1,w=3]：Y 为 [31,39,47]。", 17, BLACK)
    f.txt(24, 663, "同一地址的 R 却是 [45,46,47]，混入了不同像素。", 17, BLACK)
    f.txt(24, 706, "47在末角，碰巧相同；41的坐标才能揭穿这个错误。", 17, BLACK)
    out["01-4d-permute"] = f.d

    f = _figure(Figure, "沿通道广播：三个偏置服务全部48个格",
                "Z = X + b.view(1,3,1,1)；B、H_img、W_img轴共享同一偏置。", 730)
    for c, bias in enumerate((10, 100, 1000)):
        x = 101 + 191 * c
        f.txt(x + 58, 91, "c=" + str(c), 16, BLACK, "middle")
        _grid(f, x, 104, [[bias]], cell_w=116, cell_h=43, size=20)
    f.txt(24, 180, "[B=2,C=3,H=2,W=4] + [1,3,1,1] → [2,3,2,4]", 17, BLACK)
    for b in range(2):
        y = 224 + b * 200
        f.txt(24, y - 13, "输出样本 b=" + str(b), 18, BLACK)
        for c, bias in enumerate((10, 100, 1000)):
            values = [[_value(b, c, h, w) + bias for w in range(4)] for h in range(2)]
            _slice_panel(f, 24 + c * 222, y, "c=" + str(c) + "；每格 +" + str(bias),
                         values, marks={(1, 3)} if (b, c) == (1, 2) else (), size=16)
    f.txt(24, 632, "黑底格：Z[1,2,1,3] = 47 + 1000 = 1047。", 18, BLACK)
    f.txt(24, 668, "偏置只存3个参数；不是为每个样本或像素再学一份。", 17, BLACK)
    f.txt(24, 702, "单独 [3] 会右对齐到W；本例W=4，因此直接 X+b 报错。", 16, BLACK)
    out["01-4d-broadcast"] = f.d

    f = _figure(Figure, "沿哪根轴求平均，就汇总哪一组数",
                "同一个 X[2,3,2,4]；下面两种操作都保留B=2，且使用keepdim。", 730)
    f.txt(24, 96, "① 沿C平均：[2,3,2,4] → [2,1,2,4]", 19, BLACK)
    f.txt(24, 129, "同一像素跨通道读取：X[1,:,1,3] = [31,39,47]，均值39。", 16, BLACK)
    for b in range(2):
        x = 99 + b * 317
        f.txt(x + 84, 166, "b=" + str(b) + "，C已收成1", 17, BLACK, "middle")
        for w in range(4):
            f.txt(x + 21 + w * 42, 194, "w" + str(w), 16, BLACK, "middle")
        for h in range(2):
            f.txt(x - 10, 233 + h * 42, "h" + str(h), 16, BLACK, "end")
        values = [[sum(_value(b, c, h, w) for c in range(3)) // 3
                   for w in range(4)] for h in range(2)]
        _grid(f, x, 206, values, cell_w=42, cell_h=42,
              marks={(1, 3)} if b == 1 else ())
    f.txt(24, 325, "输出的C轴长1：表示三个通道的统计量，不是取原c=0。", 16, BLACK)
    f.line(24, 350, 676, 350, BLACK)
    f.txt(24, 388, "② 沿H、W平均：[2,3,2,4] → [2,3,1,1]", 19, BLACK)
    f.txt(24, 422, "b=1、c=2的整片40…47：总和348，除以8得到43.5。", 16, BLACK)
    for c in range(3):
        f.txt(215 + c * 170, 460, "c=" + str(c), 17, BLACK, "middle")
    for b in range(2):
        f.txt(79, 504 + b * 59, "b=" + str(b), 18, BLACK, "middle")
        values = [[sum(_value(b, c, h, w) for h in range(2) for w in range(4)) / 8
                   for c in range(3)]]
        _grid(f, 130, 474 + b * 59, values, cell_w=170, cell_h=48,
              marks={(0, 2)} if b == 1 else (), size=22)
    f.txt(24, 622, "每个输出格对应一个 [1,1] 空间切片；两个样本、三通道仍在。", 16, BLACK)
    f.txt(24, 660, "keepdim保留被归约的轴，长度改成1，便于广播回原形状。", 16, BLACK)
    f.txt(24, 696, "① 每个像素汇总3个通道；② 每个通道汇总8个像素。", 17, BLACK)
    out["01-4d-reduce"] = f.d

    return out
