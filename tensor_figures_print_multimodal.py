"""Static, numbered tensor storyboards for black-and-white print.

The shared builder supplies Figure. Each cell remains interpretable without
color: position labels, borders, step numbers and hatching carry the meaning.
"""

INK = '#171717'
GRAY = '#737373'
LIGHT = '#F2F2F2'


def _grid(f, x, y, values, cell=38, height=None, hatch=None):
    height = cell if height is None else height
    for r, row in enumerate(values):
        for c, value in enumerate(row):
            xx, yy = x + c*cell, y + r*height
            f.rect(xx, yy, cell-2, height-2, '#FFFFFF', stroke=GRAY, radius=0)
            if hatch and hatch(r, c):
                # Short corner slashes leave the central numeral readable.
                f.line(xx+3, yy+12, xx+12, yy+3, GRAY, width=1.2)
                f.line(xx+cell-14, yy+height-5,
                       xx+cell-5, yy+height-14, GRAY, width=1.2)
            f.txt(xx+(cell-2)/2, yy+height*.67, str(value), 16, INK, 'middle')


def _cols(f, x, y, labels, cell):
    for c, label in enumerate(labels):
        f.txt(x+c*cell+(cell-2)/2, y, str(label), 16, INK, 'middle')


def _rows(f, x, y, labels, cell):
    for r, label in enumerate(labels):
        f.txt(x, y+r*cell+cell*.67, str(label), 16, INK, 'end')


def _box(f, x, y, w, h):
    f.rect(x, y, w, h, '#FFFFFF', stroke=GRAY, radius=3)


def _outline(f, x, y, w, h):
    for a,b,c,d in [(x,y,x+w,y),(x+w,y,x+w,y+h),
                    (x+w,y+h,x,y+h),(x,y+h,x,y)]:
        f.line(a,b,c,d,INK,width=2.5)


def build(Figure):
    out = {}

    f = Figure('四步纸面推演：从patch到向量', height=760)
    f.txt(24,58,'单通道像素值1—16；patch边长2，位置索引从0开始',16,INK)
    _box(f,24,84,314,304); _box(f,362,84,314,304)
    _box(f,24,412,314,324); _box(f,362,412,314,324)

    f.txt(42,113,'1 选出左上角的p0',18,INK)
    _cols(f,88,139,[0,1,2,3],38)
    _rows(f,78,150,[0,1,2,3],38)
    _grid(f,88,150,[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    _outline(f,87,149,76,76)
    f.txt(42,329,'粗框内四格来自两条像素行。',16,INK)
    f.txt(42,358,'p0坐标：(0,0),(0,1),(1,0),(1,1)',16,INK)

    f.txt(380,113,'2 按固定顺序铺成一行',18,INK)
    f.txt(410,141,'p0的2×2格子',16,INK)
    _grid(f,410,153,[[1,2],[5,6]])
    _outline(f,409,152,76,76)
    f.line(492,190,648,190,INK)
    f.line(648,190,648,300,INK)
    f.arrow(648,300,627,300,INK)
    _cols(f,388,260,['u0','u1','u2','u3'],58)
    _grid(f,388,272,[[1,2,5,6]],58)
    f.txt(380,358,'只改位置，数值保持不变。',16,INK)

    f.txt(42,441,'3 这同一行右乘W',18,INK)
    _grid(f,42,464,[[1,2,5,6]],46,height=38)
    f.txt(250,489,'[1,4]',16,INK)
    f.txt(76,536,'W [4,2]',16,INK)
    _grid(f,76,548,[[1,0],[0,1],[1,0],[0,1]],36)
    f.txt(173,587,'p0 @ W =',16,INK)
    _grid(f,204,610,[[6,8]],45)
    f.arrow(151,628,195,628,INK)
    f.txt(42,717,'6 = 1+5；8 = 2+6',16,INK)

    f.txt(380,441,'4 四块分别投影，按行汇集',18,INK)
    _cols(f,470,483,['D0','D1'],44)
    _rows(f,457,495,['p0','p1','p2','p3'],44)
    _grid(f,470,495,[[6,8],[10,12],[22,24],[26,28]],44)
    f.txt(380,699,'保留4个位置，通道宽度为2。',16,INK)
    f.txt(380,723,'各行不相加；重排不是训练。',16,INK)
    f.arrow(341,235,359,235,INK)
    f.arrow(341,575,359,575,INK)
    out['12-print-patch-steps'] = f.d

    f = Figure('四步纸面推演：缺失声音的掩码', height=760)
    f.txt(24,58,'特例：仅声音缺失，其他模态均有效；黑白斜线表示禁读键',16,INK)
    f.txt(24,91,'1 固定账本：34个物理源位置',18,INK)
    f.txt(24,119,'CLS 0｜文字1—5｜图6—9｜音10—17｜视频18—33',16,INK)
    _grid(f,80,136,[list(range(17))],30)
    _grid(f,80,178,[list(range(17,34))],30)
    f.txt(24,158,'T→',16,INK)
    f.txt(80,231,'折成两行展示；源序列仍有34个位置。',16,INK)
    f.line(24,246,676,246,GRAY,width=1)

    f.txt(24,273,'2 只禁读10—17：key mask = 1',18,INK)
    for row in range(2):
        start = row*17; y = 302+53*row
        _cols(f,80,y-10,range(start,start+17),30)
        vals = [[int(10 <= start+i <= 17) for i in range(17)]]
        _grid(f,80,y,vals,30,hatch=lambda r,c,s=start:10<=s+c<=17)
    f.txt(80,412,'1=禁止读取这个键；这8行没有被删除。',16,INK)
    f.line(24,426,676,426,GRAY,width=1)

    f.txt(24,453,'3 手算特例：34个原分数全为0',18,INK)
    f.txt(24,480,'遮罩前：每列1/34。遮罩后：只在26个可读列归一化。',16,INK)
    _cols(f,165,505,[0,'…',9,10,'…',17,18,'…',33],50)
    f.txt(145,541,'加mask后',16,INK,'end')
    _grid(f,165,517,[[0,'…',0,'−∞','…','−∞',0,'…',0]],50,height=36,
          hatch=lambda r,c:3<=c<=5)
    f.txt(145,584,'softmax后',16,INK,'end')
    _grid(f,165,560,[['1/26','…','1/26',0,'…',0,'1/26','…','1/26']],50,height=36,
          hatch=lambda r,c:3<=c<=5)
    f.txt(80,621,'26×(1/26)+8×0=1；这是算例，不是真实模型权重。',16,INK)
    f.line(24,634,676,634,GRAY,width=1)

    f.txt(24,660,'4 memory仍为[34,64]，缺失行不保证是0',18,INK)
    f.lines(24,692,['键mask不负责清零查询输出；后续读取仍须使用mask。',
                    'CLS 0始终有效，至少保留一个可读键，避免全屏蔽。'],16,INK,27)
    out['17-print-missing-mask'] = f.d
    return out
