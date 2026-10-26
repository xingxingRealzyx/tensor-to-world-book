"""Four-dimensional image tensors drawn as indexed, printable slices.

The shared builder supplies Figure. All values are either the independently
checked arange(48) example or explicitly symbolic indices, never activations
claimed to come from the trained TinyOmni checkpoint.
"""

INK = '#161616'
GRAY = '#777777'


def _grid(f,x,y,values,cell=38,height=None):
    height = cell if height is None else height
    for r,row in enumerate(values):
        for c,value in enumerate(row):
            xx,yy=x+c*cell,y+r*height
            f.rect(xx,yy,cell-2,height-2,'#FFFFFF',stroke=GRAY,radius=0)
            f.txt(xx+(cell-2)/2,yy+height*.66,str(value),16,INK,'middle')


def _cols(f,x,y,labels,cell=38):
    for c,label in enumerate(labels):
        f.txt(x+c*cell+(cell-2)/2,y,str(label),16,INK,'middle')


def _rows(f,x,y,labels,height=38):
    for r,label in enumerate(labels):
        f.txt(x,y+r*height+height*.66,str(label),16,INK,'end')


def _outline(f,x,y,w,h):
    for a,b,c,d in [(x,y,x+w,y),(x+w,y,x+w,y+h),
                    (x+w,y+h,x,y+h),(x,y+h,x,y)]:
        f.line(a,b,c,d,INK,width=2.3)


def _new(Figure,title,subtitle,height=760):
    f=Figure(title,height=height)
    f.txt(24,58,subtitle,16,INK)
    return f


def build(Figure):
    out={}

    f=_new(Figure,'1×1卷积沿通道取数，保留空间',
           'X:[2,3,2,4]；只展开b=1，固定像素(y=1,x=3)')
    for c,x in enumerate([54,268,482]):
        f.txt(x,94,f'b=1，c={c}：H_img↓',16,INK)
        _cols(f,x,117,[0,1,2,3])
        _rows(f,x-8,130,[0,1])
        values=[[24+c*8+r*4+j for j in range(4)] for r in range(2)]
        _grid(f,x,130,values)
        _outline(f,x+3*38-1,167,38,38)
        f.arrow(x+3*38+18,209,238+62*c,246,INK)
    f.txt(24,305,'同一像素的C向量',16,INK)
    _cols(f,208,272,['c0','c1','c2'],62)
    _grid(f,208,284,[[31,39,47]],62,height=46)
    f.txt(430,308,'跨C取数，不移动像素。',16,INK)

    f.txt(86,369,'M逐行点积：o↓，c→',16,INK)
    _cols(f,86,390,[0,1,2],52)
    _rows(f,78,403,['o0','o1'],42)
    _grid(f,86,403,[[1,-2,'.5'],['-.5',1,2]],52,height=42)
    f.txt(280,369,'点积结果',16,INK)
    _grid(f,280,403,[['-23.5'],['117.5']],74,height=42)
    f.arrow(300,333,300,348,INK)
    f.arrow(248,445,269,445,INK)
    f.txt(398,369,'偏置β',16,INK)
    _grid(f,398,403,[['.25'],['-.75']],62,height=42)
    f.txt(540,369,'加偏置后',16,INK)
    _grid(f,540,403,[['-23.25'],['116.75']],104,height=42)
    f.txt(375,447,'+',20,INK)
    f.arrow(466,445,528,445,INK)
    f.txt(24,523,'o0：31×1 + 39×(−2) + 47×0.5 + 0.25 = −23.25',16,INK)
    f.txt(24,550,'o1：31×(−0.5) + 39×1 + 47×2 − 0.75 = 116.75',16,INK)

    f.txt(41,587,'b1,o0：H_img↓，W_img→',16,INK)
    f.txt(372,587,'b1,o1：H_img↓，W_img→',16,INK)
    _grid(f,41,602,[['-19.75','-20.25','-20.75','-21.25'],
                    ['-21.75','-22.25','-22.75','-23.25']],68,height=42)
    _grid(f,372,602,[['99.25','101.75','104.25','106.75'],
                     ['109.25','111.75','114.25','116.75']],68,height=42)
    _outline(f,244,643,68,42);_outline(f,575,643,68,42)
    f.txt(24,729,'完整输出[2,2,2,4]：批2、输出通道2、高2、宽4；b0独立计算。',16,INK)
    out['12-4d-channelconv']=f.d

    f=_new(Figure,'matmul不会替你识别通道轴',
           '同一个X[2,3,2,4]；NCHW的最后一轴是图像宽度',740)
    f.txt(24,94,'A 直接X @ U：收缩的是W_img',18,INK)
    f.txt(44,123,'X[1,2,1,:]：固定b、c、y',16,INK)
    _cols(f,44,150,['x0','x1','x2','x3'],60)
    _grid(f,44,163,[[44,45,46,47]],60,height=48)
    _outline(f,223,162,60,48)
    f.txt(350,123,'U [4,2]',16,INK)
    _grid(f,350,138,[[1,0],[0,1],[1,0],[0,1]],38)
    f.txt(495,157,'输出这条空间行',16,INK)
    _grid(f,500,179,[[90,92]],74,height=50)
    f.arrow(289,187,336,187,INK);f.arrow(432,204,486,204,INK)
    f.txt(24,321,'结果[2,3,2,2]：C仍是3，空间宽度从4变成2。',16,INK)
    f.txt(24,349,'直接X @ M.T会报错：最后轴4，无法与通道维3收缩。',16,INK)
    f.line(24,370,676,370,GRAY,width=1)

    f.txt(24,402,'B 先permute到BHWC，再右乘M.T',18,INK)
    f.txt(44,433,'BHWC:[2,2,4,3]；取[1,1,3,:]得到通道向量',16,INK)
    _cols(f,44,459,['c0','c1','c2'],62)
    _grid(f,44,472,[[31,39,47]],62,height=48)
    _outline(f,167,471,62,48)
    f.txt(310,455,'M.T [3,2]',16,INK)
    _grid(f,310,474,[[1,'-.5'],[-2,1],['.5',2]],52,height=44)
    f.txt(486,479,'加偏置后的两通道',16,INK)
    _grid(f,486,501,[['-23.25','116.75']],92,height=54)
    f.arrow(236,496,298,496,INK);f.arrow(420,528,474,528,INK)
    f.txt(24,639,'再permute回NCHW：[2,2,2,4]，空间位置不变。',18,INK)
    f.txt(24,672,"等价：einsum('bchw,oc->bohw',X,M) + 广播偏置",16,INK)
    f.txt(24,705,'b、y、x都保留；只沿输入通道c求和。',16,INK)
    out['12-4d-matmul']=f.d

    f=_new(Figure,'patch卷积的四维输出怎样排成序列',
           '同一X[2,3,2,4]；核2×2、步长2，每个样本恰有2个patch')
    f.txt(24,93,'1 每个通道各自求块内均值',18,INK)
    for b,x,values in [(0,78,[[2.5,4.5],[10.5,12.5],[18.5,20.5]]),
                       (1,420,[[26.5,28.5],[34.5,36.5],[42.5,44.5]])]:
        f.txt(x,119,f'b={b}：C↓，块位置N→',16,INK)
        _cols(f,x,143,['p0','p1'],68);_rows(f,x-8,156,['c0','c1','c2'],38)
        _grid(f,x,156,values,68,height=38)
    f.txt(24,300,'例如b0,c0,p0：(0+1+4+5)/4 = 2.5。',16,INK)
    f.txt(24,327,'每通道2×2核都等于M[o,c]/4；最后加一次对应偏置。',16,INK)
    f.line(24,343,676,343,GRAY,width=1)
    f.txt(24,370,'2 输出Y[2,2,1,2]：批B × 通道D × 高1 × 宽2',18,INK)
    for b,x,rows in [(0,90,[[-9,-10],[45.5,50.5]]),
                      (1,430,[[-21,-22],[105.5,110.5]])]:
        for d,values in enumerate(rows):
            yy=413+d*77
            f.txt(x,yy-13,f'b={b},d={d}：一片[1,2]',16,INK)
            _grid(f,x,yy,[values],68,height=42)
    f.txt(24,560,'3 flatten(2)保留B、D；再交换D与N',18,INK)
    f.txt(24,588,'[2,2,1,2] → [2,2,2]（B,D,N）→ [2,2,2]（B,N,D）',16,INK)
    for b,x,values in [(0,120,[[-9,45.5],[-10,50.5]]),
                       (1,440,[[-21,105.5],[-22,110.5]])]:
        f.txt(x-50,619,f'b={b}',16,INK)
        _cols(f,x,619,['d0','d1'],74);_rows(f,x-8,632,['p0','p1'],38)
        _grid(f,x,632,values,74,height=38)
    _outline(f,497,489,68,42);_outline(f,513,669,74,38)
    f.line(571,511,642,511,INK);f.line(642,511,642,688,INK)
    f.arrow(642,688,596,688,INK)
    f.txt(24,739,'两个[2,2,2]的轴顺序不同；B从未并入N，不会混合两张图。',16,INK)
    out['12-4d-patchbatch']=f.d

    f=_new(Figure,'TinyOmni的四维切片怎样成为序列',
           '真实接口：[2,3,16,16] → Conv8/8 → [2,64,2,2]；格值为符号')
    for b,x,left,right in [(0,24,'a','u'),(1,362,'b','v')]:
        f.rect(x,84,314,623,'#FFFFFF',stroke=GRAY,radius=3)
        f.txt(x+16,111,f'b={b}：64个通道，只画0和63',16,INK)
        for d,xx,prefix in [(0,x+41,left),(63,x+203,right)]:
            f.txt(xx,138,f'd={d}',16,INK)
            _grid(f,xx,152,[[prefix+'0',prefix+'1'],[prefix+'2',prefix+'3']],38)
        f.txt(x+152,195,'…',20,INK)
        f.txt(x+16,255,'每个通道仍是一片2×2空间网格',16,INK)
        f.arrow(x+157,267,x+157,290,INK)
        f.txt(x+16,315,'合并高宽 → [64,4]',18,INK)
        _cols(f,x+45,340,[0,1,2,3],52)
        _rows(f,x+38,353,['d0','…','d63'],31)
        _grid(f,x+45,353,[[left+str(n) for n in range(4)],['…']*4,
                          [right+str(n) for n in range(4)]],52,height=31)
        f.arrow(x+157,453,x+157,476,INK)
        f.txt(x+16,501,'交换D、N → [4,64]',18,INK)
        _cols(f,x+45,527,['d0','…','d63'],68)
        _rows(f,x+38,540,['p0','p1','p2','p3'],34)
        _grid(f,x+45,540,[[left+str(n),'…',right+str(n)] for n in range(4)],68,height=34)
        f.txt(x+16,697,f'批次身份b={b}始终不变',16,INK)
    f.txt(24,737,'整批flatten(2)→[2,64,4]；transpose(1,2)→[2,4,64]。',16,INK)
    out['17-4d-real-layout']=f.d
    return out
