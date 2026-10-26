"""Indexed tensor diagrams for Chapters 12–17.

The caller supplies Figure; importing the shared builder here would create a
cycle. Values are explicitly labelled teaching values or pixel indices, never
claimed to be a trained model's activations.
"""

BLUE = '#235B93'
TEAL = '#087F82'
ORANGE = '#B96525'
INK = '#172D40'
MUTED = '#536D7E'
COLORS = ['#DCEAF7', '#DCF0E8', '#FFF0D8', '#EDE2F5']


def _grid(f, x, y, values, cell=36, fills=None, font=14):
    """Draw every cell explicitly so no label is silently reduced below 14pt."""
    if cell < 30 or font < 14:
        raise ValueError('Tensor cells must be >=30 and their text >=14')
    width = len(values[0])
    for r, row in enumerate(values):
        if len(row) != width:
            raise ValueError('Ragged tensor grid')
        for c, value in enumerate(row):
            fill = fills[r][c] if fills else COLORS[0]
            f.rect(x + c * cell, y + r * cell, cell - 2, cell - 2,
                   fill, stroke='#FFFFFF', radius=2)
            f.txt(x + c * cell + (cell - 2) / 2,
                  y + r * cell + cell * .65, str(value), font, INK, 'middle')


def _outline(f, x, y, w, h, color=ORANGE):
    for a, b, c, d in [(x,y,x+w,y),(x+w,y,x+w,y+h),
                       (x+w,y+h,x,y+h),(x,y+h,x,y)]:
        f.line(a,b,c,d,color,width=2.2)


def _cols(f, x, y, labels, cell=36):
    for c, label in enumerate(labels):
        f.txt(x + c * cell + (cell - 2) / 2, y, str(label), 14, MUTED, 'middle')


def _rows(f, x, y, labels, cell=36):
    for r, label in enumerate(labels):
        f.txt(x, y + r * cell + cell * .65, str(label), 14, MUTED, 'end')


def _figure(Figure, title, subtitle, height=600):
    f = Figure(title, height=height)
    f.txt(24, 58, subtitle, 14, MUTED)
    return f


def build(Figure):
    out = {}

    f = _figure(Figure, '像素怎样重排成四行patch',
                '灰度教学图；p=2，行优先排列；重排本身没有可学习参数', 600)
    pixels = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    patch = [[1,2,5,6],[3,4,7,8],[9,10,13,14],[11,12,15,16]]
    fills = [[COLORS[(r//2)*2+c//2] for c in range(4)] for r in range(4)]
    f.txt(46, 93, '像素：H↓，W→', 15, BLUE)
    _cols(f,46,117,range(4),36); _rows(f,39,130,range(4),36)
    _grid(f,46,130,pixels,36,fills)
    _outline(f,45,129,71,71)
    f.txt(294,93,'patch序列：T↓，块内u→',15,TEAL)
    _cols(f,294,117,range(4),36); _rows(f,287,130,['p0','p1','p2','p3'],36)
    _grid(f,294,130,patch,36,[[COLORS[r]]*4 for r in range(4)])
    f.arrow(195,163,265,147,ORANGE)
    f.arrow(195,236,265,255,TEAL)
    f.txt(211,190,'切块',14,MUTED); f.txt(211,213,'重排',14,MUTED)
    f.txt(548,93,'输出：T↓，D→',15,ORANGE)
    _cols(f,548,117,[0,1],46)
    _grid(f,548,130,[[6,8],[10,12],[22,24],[26,28]],46,
          [[COLORS[r]]*2 for r in range(4)])
    f.arrow(442,200,535,200,TEAL); f.txt(466,180,'× W',16,TEAL)
    f.txt(46,343,'投影矩阵 W：u↓，D→',15,BLUE)
    _grid(f,46,361,[[1,0],[0,1],[1,0],[0,1]],36)
    f.lines(160,387,['p0 = [1, 2, 5, 6]',
                     'p0 W = [1+5, 2+6] = [6, 8]',
                     '同一张 W 作用于全部四行，不混合 T 轴。'],16,INK,36)
    f.txt(24,556,'扩大到RGB时还要保留通道轴C；块内排列必须与投影权重一致。',15,MUTED)
    out['12-tensor-patches'] = f.d

    f = _figure(Figure, '视觉与文本沿位置轴拼接',
                '固定一个样本b；下面D=3、视觉4行、文本2行均为教学缩小值', 560)
    vision = [[1,0,2],[0,2,1],[2,1,0],[1,1,1]]
    text = [[9,8,7],[6,5,4]]
    f.txt(54,94,'视觉 V [4,3]：D→',15,BLUE)
    _cols(f,54,118,range(3),40); _rows(f,46,130,['v0','v1','v2','v3'],40)
    _grid(f,54,130,vision,40,[[COLORS[0]]*3]*4)
    f.txt(54,326,'文本 E [2,3]：D→',15,TEAL)
    _cols(f,54,350,range(3),40); _rows(f,46,362,['t0','t1'],40)
    _grid(f,54,362,text,40,[[COLORS[1]]*3]*2)
    f.txt(410,94,'X [6,3]：T↓，D→',15,ORANGE)
    _cols(f,410,118,range(3),40)
    _grid(f,410,130,vision+text,40,[[COLORS[0]]*3]*4+[[COLORS[1]]*3]*2)
    _rows(f,402,130,['0:v0','1:v1','2:v2','3:v3','4:t0','5:t1'],40)
    for i in range(4):
        f.arrow(178,150+40*i,356,150+40*i,BLUE)
    for i in range(2):
        f.arrow(178,382+40*i,356,310+40*i,TEAL)
    f.lines(54,479,['concat(dim=T)：4行 + 2行 = 6行；通道宽度仍为3。',
                     '不是相加，也不是把两种模态拼成更长的D轴。'],16,INK,30)
    out['13-tensor-concat'] = f.d

    f = _figure(Figure, '交叉注意力保留查询位置轴',
                '两条文本查询，三个视觉键；数值为单头教学例，d=2', 620)
    f.txt(42,93,'Q：文本q↓，d→',15,BLUE)
    _cols(f,42,117,[0,1],44); _rows(f,34,130,[0,1],44)
    _grid(f,42,130,[[1,0],[0,1]],44)
    f.txt(238,93,'K.T：d↓，视觉j→',15,TEAL)
    _cols(f,238,117,['j0','j1','j2'],44)
    _grid(f,238,130,[[1,0,1],[0,1,1]],44)
    f.txt(486,93,'S = Q K.T',15,ORANGE)
    _cols(f,486,117,['j0','j1','j2'],44)
    _grid(f,486,130,[[1,0,1],[0,1,1]],44)
    f.arrow(135,174,226,174); f.arrow(374,174,474,174)
    f.line(552,220,552,277,ORANGE); f.line(552,277,105,277,ORANGE)
    f.arrow(105,277,105,344,ORANGE)
    f.txt(208,266,'除以√2，再沿视觉键轴做softmax',14,ORANGE)
    f.txt(42,330,'A：q↓，j→',15,BLUE)
    _cols(f,42,355,['j0','j1','j2'],44)
    _grid(f,42,368,[['.401','.198','.401'],['.198','.401','.401']],44)
    f.txt(280,330,'V：j↓，d_v→',15,TEAL)
    _cols(f,280,355,[0,1],44); _rows(f,272,368,['j0','j1','j2'],44)
    _grid(f,280,368,[[2,0],[0,2],[1,1]],44)
    f.txt(504,330,'O = A V',15,ORANGE)
    _cols(f,504,355,[0,1],58); _rows(f,496,368,['q0','q1'],58)
    _grid(f,504,368,[['1.203','.797'],['.797','1.203']],58)
    f.arrow(177,412,235,412); f.arrow(372,412,471,412)
    f.lines(24,550,['输出有两行，对应q0、q1；三个视觉位置沿j轴加权求和。',
                     'q0的第一维 = .401×2 + .198×0 + .401×1 ≈ 1.203。'],15,INK,29)
    out['13-tensor-cross'] = f.d

    f = _figure(Figure, '声音切帧后仍要保留时间坐标',
                '上半：8点小算例切成2帧；下半：不同采样时刻的特征示意', 640)
    f.txt(42,92,'波形：采样点s→',15,BLUE)
    _cols(f,42,117,range(8),38)
    _grid(f,42,130,[[0,1,0,-1,0,2,0,-2]],38,
          [[COLORS[0]]*4+[COLORS[1]]*4])
    _outline(f,41,129,151,37,BLUE); _outline(f,193,129,151,37,TEAL)
    f.txt(456,92,'帧j↓，帧内u→',15,TEAL)
    _cols(f,456,117,range(4),38); _rows(f,448,130,['0','1'],38)
    _grid(f,456,130,[[0,1,0,-1],[0,2,0,-2]],38,
          [[COLORS[0]]*4,[COLORS[1]]*4])
    f.arrow(348,148,433,148,BLUE)
    f.txt(355,188,'长度4',14,MUTED); f.txt(355,211,'步长4',14,MUTED)
    f.txt(42,245,'本例：frame[j,u] = wave[4j+u]；重叠窗口会重复读取采样点。',15,INK)
    f.txt(95,310,'音频特征：时间↓，D→',15,BLUE)
    _cols(f,95,338,[0,1,2],40)
    _rows(f,84,352,['0.00','0.02','0.04','0.06'],40)
    _grid(f,95,352,[['a00','a01','a02'],['a10','a11','a12'],
                     ['a20','a21','a22'],['a30','a31','a32']],40)
    f.txt(455,310,'视频特征：时间↓，D→',15,TEAL)
    _cols(f,455,338,[0,1,2],40)
    _grid(f,455,352,[['v00','v01','v02']],40,[[COLORS[1]]*3])
    _grid(f,455,432,[['v10','v11','v12']],40,[[COLORS[1]]*3])
    f.txt(447,378,'0.00',14,MUTED,'end'); f.txt(447,458,'0.04',14,MUTED,'end')
    f.arrow(219,372,392,372,TEAL); f.arrow(219,452,392,452,TEAL)
    f.txt(254,402,'按时间对齐',15,TEAL)
    f.lines(24,562,['音频第2行与视频第1行同在0.04秒，不应只按行号配对。',
                     'D列是特征通道；它既不是时间，也不是音频采样率。'],15,INK,30)
    out['14-tensor-time'] = f.d

    f = _figure(Figure, '动作块按时间行组织七个分量',
                '固定一个样本；教学H=4、A=7，不向任何真实机器人发送命令', 610)
    actions = [['.1','0','0','0','0','0','1'],
               ['.1','.1','0','0','0','0','1'],
               ['0','0','.1','0','0','0','0'],
               ['0','0','.1','0','0','0','0']]
    f.txt(112,101,'A轴→：同一时刻的动作分量',16,BLUE)
    _cols(f,112,131,['dx','dy','dz','rx','ry','rz','夹爪'],46)
    _rows(f,100,147,['h0','h1','h2','h3'],46)
    _grid(f,112,147,actions,46,[[COLORS[2]]*7]*2+[[COLORS[0]]*7]*2)
    f.txt(24,240,'H↓',16,BLUE)
    _outline(f,110,145,322,92,ORANGE)
    f.lines(473,169,['先执行前r=2行', '共14个标量', '再观察并重规划'],15,ORANGE,29)
    f.arrow(435,191,464,191,ORANGE)
    f.txt(42,387,'若接口要求展平：先h0全部7项，再h1全部7项',15,TEAL)
    _cols(f,112,418,range(14),36)
    _grid(f,112,432,[actions[0]+actions[1]],36,
          [[COLORS[2]]*7+[COLORS[1]]*7])
    f.line(273,334,273,365,TEAL); f.line(273,365,650,365,TEAL)
    f.line(650,365,650,449,TEAL); f.arrow(650,449,622,449,TEAL)
    f.lines(24,526,['a[h,j] ↔ flat[7h+j]；先排完所有dx会改变接口含义。',
                     'H是未来动作时刻；flow生成进度是另一根轴，不能混用。'],15,INK,31)
    out['15-tensor-actions'] = f.d

    f = _figure(Figure, '批量补齐后掩码要跟着位置走',
                '双向编码器教学例：B=2、T=4、D=2；1在两种mask中含义不同', 640)
    f.txt(52,94,'b0：长度4，D→',15,BLUE)
    _cols(f,52,118,[0,1],40); _rows(f,44,130,range(4),40)
    _grid(f,52,130,[[1,2],[3,4],[5,6],[7,8]],40)
    f.txt(245,94,'b1：长度2，D→',15,TEAL)
    _cols(f,245,118,[0,1],40); _rows(f,237,130,range(4),40)
    _grid(f,245,130,[[9,8],[7,6],['P','P'],['P','P']],40,
          [[COLORS[1]]*2]*2+[['#DCE3E9']*2]*2)
    f.txt(425,130,'key_padding_mask',15,ORANGE)
    f.txt(425,153,'B↓，T→；1=禁止读键',14,ORANGE)
    _cols(f,425,180,range(4),46); _rows(f,417,192,['b0','b1'],46)
    _grid(f,425,192,[[0,0,0,0],[0,0,1,1]],46,
          [[COLORS[0]]*4,[COLORS[1]]*2+['#DCE3E9']*2])
    f.arrow(329,250,406,250,TEAL)
    f.txt(52,344,'b1注意力：查询i↓，键j→',15,BLUE)
    _cols(f,52,373,range(4),38); _rows(f,44,386,range(4),38)
    _grid(f,52,386,[['读','读','禁','禁']]*4,38,
          [[COLORS[0]]*2+['#DCE3E9']*2]*4)
    f.txt(351,344,'b1的loss_valid：T→',15,TEAL)
    _cols(f,351,373,range(4),46)
    _grid(f,351,386,[[1,1,0,0]],46,[[COLORS[1]]*2+['#DCE3E9']*2])
    f.lines(351,467,['1=参与计分，0=忽略', '只除以有效标签数量', 'P表示占位，不限定为数值0'],15,INK,28)
    f.line(563,287,563,318,ORANGE); f.line(563,318,275,318,ORANGE)
    f.line(275,318,275,405,ORANGE); f.arrow(275,405,210,405,ORANGE)
    f.txt(24,596,'键mask屏蔽整列；填充查询仍可能有输出，但不能被当作有效结果。',15,MUTED)
    out['16-tensor-padding'] = f.d

    f = _figure(Figure, 'TinyOmni的RGB块内部次序',
                '真实接口[C,H,W]=[3,16,16]；用数值索引追踪，不是照片像素值', 640)
    coords = [0,1,8,9]
    for c, x in enumerate([44,254,464]):
        f.txt(x,92,f'C={c}：'+['R','G','B'][c]+'通道',15,[BLUE,TEAL,ORANGE][c])
        _cols(f,x,117,coords,36); _rows(f,x-8,130,coords,36)
        values = [[c*256+y*16+xx for xx in coords] for y in coords]
        _grid(f,x,130,values,36,
              [[COLORS[(r//2)*2+col//2] for col in range(4)] for r in range(4)])
        _outline(f,x-1,129,71,71,ORANGE)
        f.txt(x,302,'p0向量中的槽位编号',14,MUTED)
        f.arrow(x+70,312,x+70,342,[BLUE,TEAL,ORANGE][c])
        _grid(f,x,353,[[64*c,64*c+1,'…',64*c+63]],36,
              [[COLORS[c]]*4])
    f.txt(24,420,'上方仅展示每个8×8块的左上2×2；行列标签是原图坐标。',15,INK)
    f.txt(24,451,'p0内部：先R的64项，再G的64项，再B的64项，共192项。',15,INK)
    f.txt(24,482,'卷积权重[D,C,8,8]：展平并转置，得到块乘法权重[192,D]。',15,INK)
    _grid(f,455,517,[['p0','p1'],['p2','p3']],42,
          [[COLORS[0],COLORS[1]],[COLORS[2],COLORS[3]]])
    f.lines(24,532,['Conv2d输出 [B,64,2,2]',
                     'flatten(2) → [B,64,4]',
                     'transpose(1,2) → [B,4,64]'],16,BLUE,30)
    f.arrow(540,559,588,559,TEAL)
    _grid(f,598,493,[['p0'],['p1'],['p2'],['p3']],30,
          [[COLORS[i]] for i in range(4)])
    out['17-tensor-rgb'] = f.d

    f = _figure(Figure, '八帧声音怎样进入34个源位置',
                '真实TinyOmni：每32点一帧、共8帧；每帧投影到64维', 640)
    f.txt(52,96,'audio帧j↓，帧内u→',15,BLUE)
    _cols(f,52,121,[0,1,'…',31],36); _rows(f,44,135,range(8),36)
    _grid(f,52,135,[[32*r,32*r+1,'…',32*r+31] for r in range(8)],36)
    f.txt(268,96,'D→：只画前2列',14,TEAL)
    _cols(f,268,121,[0,1],36)
    _grid(f,268,135,[[f'a{r}0',f'a{r}1'] for r in range(8)],36,
          [[COLORS[1]]*2]*8)
    for r in range(8): f.arrow(199,153+36*r,257,153+36*r,TEAL)
    f.txt(366,111,'源序列T轴上的位置',16,ORANGE)
    f.lines(366,148,['0：CLS（1行）', '1—5：文字（5行）',
                     '6—9：图像（4行）', '10—17：声音（8行）',
                     '18—33：视频（16行）'],16,INK,37)
    f.txt(366,357,'声音帧j → 源位置10+j',15,TEAL)
    f.txt(366,390,'1+5+4+8+16 = 34',17,ORANGE)
    kinds = ['C']+['T']*5+['I']*4+['A']*8+['V']*16
    palette = {'C':'#F8DDBB','T':COLORS[0],'I':COLORS[1],
               'A':COLORS[2],'V':COLORS[3]}
    for row in range(2):
        start=row*17; y=476+72*row
        _cols(f,82,y,list(range(start,start+17)),30)
        _grid(f,82,y+12,[kinds[start:start+17]],30,
              [[palette[k] for k in kinds[start:start+17]]])
    f.txt(24,612,'为排版折成两行；模型实际仍是[B,34,64]，不是两条独立序列。',14,MUTED)
    out['17-tensor-source'] = f.d

    f = _figure(Figure, '共享源状态怎样变成多种输出',
                '真实维度，示意特征值；文字读整段memory，连续头只读第0行CLS', 640)
    f.txt(40,96,'memory [34,64]',15,BLUE)
    _cols(f,48,124,[0,1,2,63],36)
    _rows(f,41,140,['0','1','2','33'],36)
    _grid(f,48,140,[['.2','-.1','.5','.3'],['h10','h11','h12','…'],
                    ['h20','h21','h22','…'],['…','…','…','…']],36,
          [[COLORS[2]]*4]+[[COLORS[0]]*4]*3)
    _outline(f,47,139,143,35,ORANGE)
    f.txt(330,94,'文字：因果解码，逐个选择token',15,BLUE)
    _grid(f,330,114,[['red','circle','right','EOS']],76,[[COLORS[0]]*4])
    f.line(192,245,217,245,BLUE);f.line(217,245,217,151,BLUE)
    f.arrow(217,151,317,151,BLUE)
    f.line(192,157,260,157,ORANGE); f.line(260,157,260,546,ORANGE)
    f.txt(358,224,'图像头：C↓，展平像素p→',15,TEAL)
    _cols(f,358,248,[0,1,'…'],36);_rows(f,350,260,['R','G','B'],36)
    _grid(f,358,260,[['r0','r1','…'],['g0','g1','…'],['b0','b1','…']],36,
          [[COLORS[1]]*3]*3)
    f.arrow(260,314,321,314,ORANGE)
    f.lines(488,283,['768项reshape', '→ [3,16,16]'],15,TEAL,28)
    f.txt(358,391,'音频头：采样轴s→，[256]',15,TEAL)
    _grid(f,358,405,[['s0','s1','s2','…','s254','s255']],44,[[COLORS[1]]*6])
    f.arrow(260,427,346,427,ORANGE)
    f.txt(358,468,'动作头：H↓，A→',15,TEAL)
    _cols(f,358,489,['x','y'],30)
    _grid(f,358,496,[['.25','0'],['.50','0'],['.75','0'],['1','0']],30,
          [[COLORS[1]]*2]*4)
    f.arrow(260,546,346,546,ORANGE)
    f.lines(40,354,['橙线仅来自CLS行', '先读条件，再输出', '不读正确答案词'],15,ORANGE,29)
    f.lines(40,468,['输入样本b保持不变', '各输出头有独立参数', '共享来源不等于复制'],15,MUTED,29)
    f.lines(460,537,['8项reshape', '→ [4,2]', '仅教学轨迹'],15,TEAL,27)
    out['17-tensor-heads'] = f.d
    return out
