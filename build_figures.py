from pathlib import Path
import math, json, wave
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Polygon, Circle
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor, Color, white

ROOT = Path(__file__).parent
pdfmetrics.registerFont(TTFont('BookSans', '/Library/Fonts/Arial Unicode.ttf'))
BLUE = '#235B93'; TEAL = '#087F82'; ORANGE = '#CC702D'
INK = '#172D40'; MUTED = '#536D7E'; PALE = '#EDF3F8'; GRAY = '#DCE3E9'

class Figure:
    def __init__(self, title, subtitle='', height=340):
        self.h=height; self.d=Drawing(700,height)
        self.rect(0,0,700,height,'#F7FAFC',stroke='#DFE8EE',radius=14)
        self.txt(24,30,title,22,INK)
        if subtitle: self.txt(24,54,subtitle,13,MUTED)
    def txt(self,x,y,s,size=15,color=INK,anchor='start'):
        self.d.add(String(x,self.h-y,s,fontName='BookSans',fontSize=size,
                          fillColor=HexColor(color),textAnchor=anchor))
    def lines(self,x,y,lines,size=15,color=INK,step=23,anchor='start'):
        for i,s in enumerate(lines): self.txt(x,y+i*step,s,size,color,anchor)
    def rect(self,x,y,w,h,fill=PALE,stroke=None,radius=7):
        self.d.add(Rect(x,self.h-y-h,w,h,rx=radius,ry=radius,
                        fillColor=HexColor(fill),strokeColor=HexColor(stroke) if stroke else None,
                        strokeWidth=1))
    def card(self,x,y,w,h,title,sub='',color=BLUE):
        self.rect(x,y,w,h,'#FFFFFF',stroke=color)
        self.txt(x+w/2,y+27,title,16,color,'middle')
        if sub: self.txt(x+w/2,y+51,sub,13,MUTED,'middle')
    def line(self,x1,y1,x2,y2,color=MUTED,dash=None,width=1.6):
        self.d.add(Line(x1,self.h-y1,x2,self.h-y2,strokeColor=HexColor(color),strokeWidth=width,strokeDashArray=dash))
    def arrow(self,x1,y1,x2,y2,color=MUTED,dash=None):
        self.line(x1,y1,x2,y2,color,dash)
        angle=math.atan2(y2-y1,x2-x1)
        pts=[x2,self.h-y2]
        for a in [angle+2.65,angle-2.65]:
            pts += [x2+8*math.cos(a),self.h-(y2+8*math.sin(a))]
        self.d.add(Polygon(pts,fillColor=HexColor(color),strokeColor=None))
    def matrix(self,x,y,rows,cols,values=None,cell=32,color=BLUE,mask=None,heat=None):
        for r in range(rows):
            for c in range(cols):
                if mask and mask(r,c): fill='#DCE3E9'
                elif heat is not None:
                    v=max(0,min(1,float(heat[r][c]))); base=HexColor(color)
                    fill=Color(1-(1-base.red)*(.1+.85*v),1-(1-base.green)*(.1+.85*v),1-(1-base.blue)*(.1+.85*v)).hexval().replace('0x','#')
                else: fill='#E0EDF8' if (r+c)%2 else '#CADFF0'
                self.rect(x+c*cell,y+r*cell,cell-2,cell-2,fill,radius=2)
                if values is not None:
                    st=str(values[r][c]); self.txt(x+c*cell+(cell-2)/2,y+r*cell+cell*.62,st,min(14,cell*.37),INK,'middle')
    def circle(self,x,y,r,color=TEAL,fill=None):
        self.d.add(Circle(x,self.h-y,r,strokeColor=HexColor(color),fillColor=HexColor(fill) if fill else None,strokeWidth=1.6))

def figures():
    out={}
    f=Figure('同一套张量语言，四个观察尺度','从离散符号，到上下文表示，再到感知和行动',330)
    cards=[('表示','整数 → 向量','01-04'),('语言','前缀 → 分布','05-08'),('稀疏与推理','选择计算与反馈','09-11'),('感知与行动','多模态 → 输出','12-16')]
    for i,(a,b,c) in enumerate(cards):
        x=25+i*171; f.card(x,105,145,98,a,b,[BLUE,TEAL,ORANGE,BLUE][i]); f.txt(x+72,184,c,13,MUTED,'middle')
        if i<3:f.arrow(x+145,153,x+166,153)
    f.lines(28,251,['每章都追问：轴是什么？信息从哪里来？输出怎样验证？',
                     '形状检查 + 数字算例 + 原始资料 + 练习'],16,INK,29)
    out['00-map']=f.d

    f=Figure('给每一根轴一个名字','B：样本  ·  T：位置  ·  D：通道',350)
    f.matrix(30,115,1,4,[[1,0,2,-1]],42)
    f.lines(30,91,['向量 [D=4]'],16,BLUE)
    f.matrix(250,115,3,4,[[1,0,2,-1],[0,1,1,0],[2,1,0,1]],36)
    f.txt(250,91,'序列 [T=3,D=4]',16,BLUE)
    f.matrix(490,96,3,4,cell=34,color=TEAL)
    f.matrix(468,120,3,4,cell=34)
    f.txt(455,275,'批量 [B=2,T=3,D=4]',16,TEAL)
    f.arrow(205,137,239,137);f.arrow(410,160,455,160)
    f.lines(30,284,['一行 = 一个位置的当前状态', '一列 = 同一表示空间的一个通道'],15,INK,25)
    out['01-tensor']=f.d

    f=Figure('相同元素，正确的轴顺序','先分头，再交换轴；直接改形状不等于转置',300)
    for x,t,s in [(24,'线性投影','[B,T,D]'),(255,'拆分通道','[B,T,H,d_h]'),(486,'交换轴','[B,H,T,d_h]')]:f.card(x,96,188,80,t,s)
    f.arrow(212,135,246,135);f.arrow(443,135,477,135)
    f.lines(28,218,['W[D,F] 混合通道；A[T,T] 混合位置。',
                    'reshape 负责分组；transpose 负责移动索引轴。'],17,INK,31)
    out['01-axes']=f.d

    f=Figure('从token编号到向量，再回到候选分数','编号是地址；向量才参与后续连续计算',345)
    f.card(25,100,130,78,'输入编号','[2,5,2]')
    f.matrix(213,80,8,4,cell=22)
    f.txt(257,282,'E[V,D]',15,BLUE,'middle')
    f.matrix(390,112,3,4,cell=31)
    f.txt(450,235,'X[T,D]',15,BLUE,'middle')
    f.arrow(155,140,205,140);f.arrow(308,140,381,140)
    f.card(546,113,130,80,'层堆叠','h[T,D]',TEAL)
    f.arrow(515,150,540,150)
    f.txt(350,314,'绑定输出权重：logits = h @ E.T，得到 [T,V]',16,ORANGE,'middle')
    out['02-embedding']=f.d

    f=Figure('注意力：先匹配位置，再传递内容','Q、K决定读取比例；V提供实际汇总的内容',355)
    f.card(24,93,130,72,'Q','[T,d_k]')
    f.card(24,194,130,72,'K.T','[d_k,T]')
    f.card(217,135,158,80,'Q @ K.T','[T,T]',TEAL)
    f.arrow(154,129,210,164);f.arrow(154,230,210,181)
    f.card(424,135,250,80,'缩放 + 掩码 + softmax','每个查询沿键轴归一化',TEAL)
    f.arrow(375,175,418,175)
    f.lines(222,271,['A[T,T] @ V_mat[T,d_v]', '→ O[T,d_v]'],19,ORANGE,29)
    f.arrow(550,215,550,256)
    out['03-attention']=f.d

    f=Figure('因果掩码与逐行权重','第i行只能读取位置0到i；对角线允许读取',350)
    f.matrix(45,105,3,3,[['可见','屏蔽','屏蔽'],['可见','可见','屏蔽'],['可见','可见','可见']],cell=56,mask=lambda r,c:c>r)
    f.txt(129,307,'可见性 [3,3]',16,BLUE,'middle')
    f.arrow(231,189,303,189)
    vals=[[1,0,0],[.3302,.6698,0],[.2483,.2483,.5035]]
    f.matrix(328,105,3,3,[['1','0','0'],['.3302','.6698','0'],['.2483','.2483','.5035']],cell=63,heat=vals)
    f.lines(550,143,['精确行和 = 1','精确行和 = 1','精确行和 = 1'],15,TEAL,63)
    f.txt(423,328,'四舍五入后的显示值；实际计算不截断',13,MUTED,'middle')
    out['03-mask']=f.d

    f=Figure('一个Pre-LN Transformer层','主路径始终保留 [B,T,D]；每个子模块写入一个增量',390)
    steps=[(35,'X'),(175,'U'),(325,'Y')]
    f.card(72,85,124,67,'LayerNorm 1','按通道归一化')
    f.card(235,85,145,67,'自注意力','位置间交流',TEAL)
    f.card(72,230,124,67,'LayerNorm 2','按通道归一化')
    f.card(235,230,145,67,'MLP','逐位置加工',TEAL)
    f.arrow(196,118,226,118);f.arrow(196,263,226,263)
    f.lines(510,119,['残差主路径','[B,T,D]'],16,BLUE,26)
    f.line(454,79,454,340,BLUE,width=2.5)
    f.circle(454,170,14,ORANGE,'#FFF3E6');f.txt(454,175,'+',18,ORANGE,'middle')
    f.circle(454,315,14,ORANGE,'#FFF3E6');f.txt(454,320,'+',18,ORANGE,'middle')
    f.arrow(380,118,440,164,TEAL);f.arrow(380,263,440,309,TEAL)
    f.line(454,79,35,79,BLUE);f.line(35,79,35,118,BLUE);f.arrow(35,118,65,118,BLUE)
    f.line(454,209,35,209,BLUE);f.line(35,209,35,263,BLUE);f.arrow(35,263,65,263,BLUE)
    f.txt(454,70,'X',16,BLUE,'middle');f.txt(477,207,'U',16,BLUE);f.txt(454,365,'Y',17,BLUE,'middle')
    f.lines(510,244,['MLP中间宽度可变', '输出必须回到D', '才可以相加'],14,MUTED,25)
    out['04-block']=f.d

    f=Figure('可见性决定结构能使用哪些信息','蓝色为可读，灰色为不可读；交叉注意力可以读取完整源表示',310)
    for x,title,mask in [(30,'编码器：双向',None),(263,'解码器：因果',lambda r,c:c>r)]:
        f.txt(x,97,title,16,BLUE);f.matrix(x+20,122,4,4,cell=32,mask=mask)
    f.txt(485,97,'编码器-解码器',16,TEAL)
    f.card(489,119,174,57,'源表示 [T_src,D]',color=TEAL)
    f.arrow(574,177,574,205,TEAL)
    f.card(489,214,174,57,'目标查询 [T_tgt,D]')
    f.txt(27,291,'每一类结构还可包含填充、分段或任务专用掩码。',14,MUTED)
    out['04-families']=f.d

    f=Figure('GPT-2 small：一条完整的前向路径','固定标本：12层、D=768、12头、V=50257、上下文1024',365)
    for x,title,sub in [(22,'token ids','[B,T]'),(250,'词嵌入 + 位置','[B,T,768]'),(478,'12 × Pre-LN块','[B,T,768]')]:f.card(x,93,200,74,title,sub)
    f.arrow(223,131,242,131);f.arrow(450,131,470,131)
    f.card(478,232,200,74,'最终LayerNorm','[B,T,768]',TEAL)
    f.card(250,232,200,74,'乘 E.T','[768,50257]',ORANGE)
    f.card(22,232,200,74,'词表logits','[B,T,50257]',ORANGE)
    f.arrow(577,168,577,224);f.arrow(477,270,459,270);f.arrow(248,270,230,270)
    f.line(350,168,350,207,BLUE,[4,3]);f.arrow(350,207,350,224,BLUE,[4,3]);f.txt(367,201,'共享E',12,BLUE)
    f.txt(25,344,'输出绑定词嵌入参数；token轴T在整条主路径上保持。',15,MUTED)
    out['05-gpt2']=f.d

    f=Figure('训练：一次矩阵前向，多个位置同时计分','输入与标签错开一位；模型不能读取各位置的未来输入',335)
    for i,a in enumerate(['a','b','c']):
        f.card(153+i*153,86,107,58,a,color=BLUE)
        f.card(153+i*153,222,107,58,['b','c','d'][i],color=ORANGE)
        f.arrow(206+i*153,145,206+i*153,214)
        f.txt(206+i*153,184,'预测',14,TEAL,'middle')
    f.txt(30,120,'输入',17,BLUE);f.txt(30,255,'目标',17,ORANGE)
    f.txt(206,310,'可见：a',14,MUTED,'middle');f.txt(359,310,'可见：a,b',14,MUTED,'middle');f.txt(512,310,'可见：a,b,c',14,MUTED,'middle')
    out['06-training']=f.d

    f=Figure('KV缓存：把历史计算留下来','每层都有缓存；生成新token时只追加它的新键和值',365)
    f.card(26,87,280,70,'Prefill：整段输入','Q,K,V 具有多个位置',BLUE)
    f.matrix(65,185,4,5,cell=24);f.txt(196,220,'历史K/V',15,BLUE)
    f.card(382,87,292,70,'Decode：一个新位置','Q [B,H,1,d_h]',TEAL)
    f.matrix(398,185,4,5,cell=24);f.rect(398,281,118,22,'#E3F3ED',stroke=TEAL,radius=2)
    f.txt(536,223,'读取历史',15,TEAL);f.txt(536,296,'追加1行',15,ORANGE)
    f.arrow(310,219,378,219)
    f.txt(27,340,'例：32个KV头 4 GiB → 8个KV头 1 GiB（详见第07章条件）',15,MUTED)
    out['07-kv']=f.d

    f=Figure('RoPE：相对角度保留位置差','同一个二维向量对，整体平移位置后点积不变',365)
    for cx,a,b,cap in [(183,60,180,'位置1与3'),(512,120,240,'位置2与4')]:
        cy=196;r=83;f.circle(cx,cy,r,'#B9C9D7');f.line(cx-r-10,cy,cx+r+10,cy,'#B9C9D7');f.line(cx,cy-r-10,cx,cy+r+10,'#B9C9D7')
        for deg,col,lab in [(a,BLUE,'q'),(b,ORANGE,'k')]:
            ex=cx+r*math.cos(math.radians(deg));ey=cy-r*math.sin(math.radians(deg));f.arrow(cx,cy,ex,ey,col);f.txt(ex+(10 if ex>=cx else -13),ey-8,lab,17,col)
        f.txt(cx,310,cap,16,INK,'middle')
    f.txt(350,346,'两边都相差120°，点积 = cos(120°) = -0.5',17,TEAL,'middle')
    out['08-rope']=f.d

    f=Figure('MoE：先分发，再按原位置合并','教学路由：3个候选专家，每个token选2个，路由缩放设为1',390)
    vals=[[2/3,1/3,0],[0,7/9,2/9],[1/4,0,3/4],[5/9,0,4/9]]
    f.matrix(26,108,4,3,[['2/3','1/3','0'],['0','7/9','2/9'],['1/4','0','3/4'],['5/9','0','4/9']],cell=43,heat=vals)
    f.txt(29,91,'4个token × 3个专家',14,BLUE)
    for i,t in enumerate(['E1：t1,t3,t4','E2：t1,t2','E3：t2,t3,t4']):
        y=85+i*83;f.card(260,y,181,61,t,color=TEAL);f.arrow(160,150+i*32,251,y+31)
        f.arrow(444,y+31,521,182)
    f.card(528,147,148,75,'加权合并','回到4个位置',ORANGE)
    f.card(260,320,182,51,'共享专家：全部token',color=BLUE)
    f.line(95,281,95,345,BLUE);f.arrow(95,345,252,345,BLUE);f.arrow(444,346,593,229,BLUE)
    out['09-moe']=f.d

    f=Figure('MLA：缓存潜变量，而不是逐头展开的KV','内容与位置分开；示意省略归一化等实现细节',370)
    f.card(22,95,135,68,'X','[B,T,D]')
    f.card(232,82,188,70,'内容潜变量 C','[B,T,r]',TEAL)
    f.card(232,210,188,70,'位置键 K_R','[B,T,p]',ORANGE)
    f.arrow(157,126,223,116);f.arrow(157,144,223,244)
    f.rect(475,78,197,211,'#FFFFFF',stroke=BLUE)
    f.lines(574,111,['每层缓存','C + K_R','每位置 r+p 个数'],17,BLUE,31,'middle')
    f.arrow(420,116,468,116,TEAL);f.arrow(420,245,468,245,ORANGE)
    f.lines(26,320,['逻辑展开：C → K_C、V；高效解码可将上投影吸收到查询/输出。',
                      '必须保留位置分支；直接把普通RoPE塞进内容压缩会破坏吸收条件。'],14,MUTED,25)
    out['10-mla']=f.d

    f=Figure('GRPO：比较同一道题的多次尝试','这是训练中的组内相对信号；推理时不自动执行参数更新',340)
    f.card(24,96,136,76,'同一道题','采样4条回答')
    f.card(215,96,170,76,'验证器奖励','[1,0,0,0]',ORANGE)
    f.card(440,96,234,76,'组内标准化','均值 .25，标准差 .433',TEAL)
    f.arrow(162,133,208,133);f.arrow(386,133,433,133)
    f.card(307,224,366,76,'优势 A','[1.732, -.577, -.577, -.577]',TEAL)
    f.arrow(558,172,558,215)
    f.card(24,224,208,76,'带约束的策略更新','裁剪、参考策略等')
    f.arrow(304,263,240,263)
    out['11-grpo']=f.d

    f=Figure('图像变成序列：切块、展平、投影','224×224×3，patch边长16；数字是教学配置',345)
    f.matrix(25,95,14,14,cell=12)
    f.txt(109,292,'14×14 = 196块',16,BLUE,'middle')
    f.arrow(199,177,247,177)
    f.card(256,136,178,82,'展平每个patch','16×16×3 = 768')
    f.arrow(437,177,480,177)
    f.card(489,136,183,82,'线性投影','[B,196,D]',TEAL)
    f.txt(260,273,'可选CLS token → 长度197',15,MUTED)
    f.txt(260,309,'网格坐标与后续位置机制仍需保留。',14,MUTED)
    out['12-patches']=f.d

    f=Figure('CLIP：图与文之间的配对学习','3个图像、3段文字；对角线是教学数据中的正确配对',355)
    vals=[[3,1,0],[0,3,1],[1,0,3]]
    f.matrix(250,110,3,3,vals,cell=60,heat=[[v/3 for v in r] for r in vals])
    for i in range(3):f.txt(280+60*i,94,f'文{i+1}',14,TEAL,'middle');f.txt(229,146+60*i,f'图{i+1}',14,BLUE,'end')
    f.arrow(447,145,580,145,TEAL);f.txt(590,150,'行softmax',14,TEAL)
    f.arrow(279,299,279,326,ORANGE);f.txt(322,330,'列softmax也训练',14,ORANGE)
    f.lines(28,160,['对角线拉近', '非配对样本区分', '不直接生成描述'],15,MUTED,29)
    out['12-clip']=f.d

    f=Figure('VLM：将视觉表示接入语言计算','投影对齐的是通道接口，不保证语义已经对齐',390)
    f.card(24,92,198,80,'视觉编码器','[B,196,768]')
    f.card(258,92,200,80,'投影器','[B,196,2048]',TEAL)
    f.card(496,92,180,80,'文本嵌入','[B,32,2048]')
    f.arrow(223,131,249,131)
    f.card(258,230,262,78,'序列拼接 → 语言模型','[B,228,2048]',ORANGE)
    f.arrow(358,173,358,222,TEAL);f.arrow(586,174,491,224,BLUE)
    f.lines(25,349,['另一类接口：文本提供Q，视觉表示提供K/V，做交叉注意力。',
                     '本图是教学组合，不代表所有VLM或GPT-4o的固定内部结构。'],14,MUTED,24)
    out['13-vlm']=f.d

    f=Figure('从语音流水线到联合多模态建模','通用架构对照；不是GPT-4o内部结构图',400)
    for x,t,s in [(25,'ASR','声音 → 文字'),(260,'文字LLM','文字 → 文字'),(495,'TTS','文字 → 声音')]:f.card(x,94,180,78,t,s)
    f.arrow(207,133,252,133);f.arrow(442,133,487,133)
    f.txt(27,209,'中间仅保留文字时，语气、节奏等信息可能损失。',15,MUTED)
    f.card(25,257,180,80,'多模态输入','音频 / 图像 / 文字')
    f.card(260,257,180,80,'联合表示与训练','跨模态信息交换',TEAL)
    f.card(495,257,180,80,'多模态输出','文字 / 音频等',ORANGE)
    f.arrow(207,297,252,297);f.arrow(442,297,487,297)
    f.txt(27,375,'模型的能力、接口开放范围、产品部署范围，需要分别核对。',15,MUTED)
    out['14-omni']=f.d

    f=Figure('VLA：动作是一段数值序列','预测K步，执行前r步，再观察；本图不运行真实机器人',400)
    f.card(24,95,185,82,'观测与任务','图像 + 语言 + 状态')
    f.card(258,95,180,82,'VLA策略','上下文 → 动作块',TEAL)
    f.card(488,95,188,82,'动作输出','[B,K,7]',ORANGE)
    f.arrow(211,135,249,135);f.arrow(439,135,480,135)
    f.line(580,178,580,227,ORANGE);f.line(580,227,112,227,ORANGE);f.arrow(112,227,112,182,ORANGE)
    f.txt(350,216,'执行前r步 → 新观测与时间戳',15,ORANGE,'middle')
    f.matrix(60,278,3,7,cell=21,color=BLUE)
    f.arrow(223,310,322,310,TEAL);f.txt(272,288,'有限步更新',14,TEAL,'middle')
    f.matrix(346,278,3,7,cell=21,color=TEAL)
    f.lines(525,294,['噪声动作块', '→ 条件动作块'],15,MUTED,28)
    f.txt(27,377,'7维仅是示例：平移3 + 旋转3 + 夹爪1；坐标、单位必须定义。',14,MUTED)
    out['15-vla']=f.d

    f=Figure('端到端系统：每个接口都需要可验证的约定','模型正确连接，并不自动意味着任务可靠完成',345)
    titles=[('采集与对齐','时间戳 / 坐标'),('表示与决策','形状 / 模型版本'),('输出与执行','schema / 单位')]
    for i,(a,b) in enumerate(titles):
        x=25+229*i;f.card(x,107,194,82,a,b,[BLUE,TEAL,ORANGE][i]);
        if i<2:f.arrow(x+196,148,x+219,148)
    f.card(253,246,210,64,'验证、日志与反馈','分层评测、错误归因',TEAL)
    f.line(580,190,580,277,ORANGE);f.arrow(580,277,469,277,ORANGE)
    f.line(252,277,120,277,TEAL);f.arrow(120,277,120,196,TEAL)
    out['16-system']=f.d

    f=Figure('TinyOmni：把多种条件写进同一个表示空间','实际默认结构：352,281参数，编码器2层，文字解码器2层',440)
    for i,(title,sub) in enumerate([('文字','5 tokens'),('图像','4 patches'),('音频','8 frames'),('视频','16 patches')]):
        x=20+i*172;f.card(x,91,145,67,title,sub)
        f.line(x+72,159,x+72,187,BLUE)
    f.line(92,187,608,187,BLUE);f.arrow(350,188,350,210,BLUE)
    f.card(216,219,268,77,'共享编码器 + 条件CLS','[B,34,64]，缺失模态被屏蔽',TEAL)
    f.txt(502,207,'+ 1 CLS',13,TEAL)
    f.line(350,297,350,318,TEAL);f.line(92,318,608,318,TEAL)
    for i,(title,sub) in enumerate([('因果文字decoder','[B,4,17] logits'),('条件图像头','[B,3,16,16]'),('条件音频头','[B,256]'),('条件动作头','[B,4,2]')]):
        x=20+i*172;f.arrow(x+72,319,x+72,336,TEAL);f.card(x,343,145,66,title,sub,ORANGE)
    f.txt(24,431,'图、音、动作头仅读取源条件，不接收teacher forcing答案。',14,MUTED)
    out['17-tinyomni']=f.d

    metrics_path=ROOT/'runs/tinyomni-final/metrics.json'
    if metrics_path.exists():
        metrics=json.loads(metrics_path.read_text());history=metrics['history']
        f=Figure('真实训练曲线：1,000次参数更新','CPU实测，seed42，batch32；纵轴为log10尺度',360)
        x0,y0=70,291;pw,ph=590,195
        for exponent in [1,0,-1,-2,-3]:
            yy=y0-(exponent+3)/4*ph
            f.line(x0,yy,x0+pw,yy,'#DCE5EB',width=.7)
            f.txt(x0-12,yy+4,f'10^{exponent}',12,MUTED,'end')
        f.line(x0,90,x0,y0,INK);f.line(x0,y0,x0+pw,y0,INK)
        points=[]
        for row in history:
            xx=x0+row['step']/1000*pw
            yy=y0-(max(-3,min(1,math.log10(max(row['total'],1e-8))))+3)/4*ph
            points.append((xx,yy))
        for a,b in zip(points,points[1:]):f.line(*a,*b,TEAL,width=.8)
        for step in [0,250,500,750,1000]:f.txt(x0+step/1000*pw,y0+20,str(step),12,MUTED,'middle')
        f.txt(350,344,f"总loss：首步 {history[0]['total']:.4f}  /  末步 {history[-1]['total']:.4f}",16,INK,'middle')
        out['17-loss']=f.d

    demo_root=ROOT/'runs/tinyomni-final'
    if (demo_root/'demo-audio/generated_image.png').exists():
        from PIL import Image
        import struct
        f=Figure('模型生成样例：不同输入，重建同一小世界','checkpoint第1000步，seed90001；图像像素放大显示',515)
        def pixels(path,x,y,size=67):
            with Image.open(path) as im:
                im=im.convert('RGB');pix=im.load();unit=size/16
                for r in range(16):
                    for c in range(16):
                        rgb=pix[c,r];col='#%02x%02x%02x'%rgb
                        f.rect(x+c*unit,y+r*unit,unit+.12,unit+.12,col,radius=0)
        def wav(path):
            with wave.open(str(path),'rb') as w:return [v/32767 for v in struct.unpack('<256h',w.readframes(256))]
        def plot(samples,x,y,width,height,color):
            pts=[(x+i/(len(samples)-1)*width,y-v*height/2) for i,v in enumerate(samples)]
            for a,b in zip(pts,pts[1:]):f.line(*a,*b,color,width=1)
        f.txt(91,85,'输入条件',14,BLUE,'middle');f.txt(314,85,'预测图像',14,TEAL,'middle')
        f.txt(436,85,'目标图像',14,MUTED,'middle');f.txt(582,85,'预测文字',14,ORANGE,'middle')
        for i,task in enumerate(['text','image','audio']):
            yy=108+i*92;folder=demo_root/f'demo-{task}'
            if task=='text':f.lines(26,yy+25,['red circle','right'],15,BLUE,23)
            elif task=='image':pixels(folder/'input_image.png',58,yy)
            else:
                f.line(25,yy+35,195,yy+35,'#D5E1E8',width=.7)
                plot(wav(folder/'input_audio.wav'),25,yy+35,170,65,BLUE)
            f.arrow(208,yy+34,260,yy+34,TEAL)
            pixels(folder/'generated_image.png',280,yy);pixels(folder/'target_image.png',402,yy)
            pred=json.loads((folder/'prediction.json').read_text())['predicted_tokens']
            f.lines(582,yy+24,[' '.join(pred[:2]),pred[2]],14,ORANGE,24,'middle')
        f.txt(28,410,'音频任务：预测波形与规范目标波形',15,INK)
        f.line(28,458,673,458,'#CCD9E2',width=.7)
        plot(wav(demo_root/'demo-audio/target_audio.wav'),28,458,645,66,'#EAB779')
        plot(wav(demo_root/'demo-audio/generated_audio.wav'),28,458,645,66,TEAL)
        f.txt(350,505,'橙：目标  /  绿：模型输出；这是合成音调，不是自然语音。',13,MUTED,'middle')
        out['17-generated']=f.d
    return out

if __name__=='__main__':
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    for name,d in figures().items():
        p=folder/(name+'.svg');renderSVG.drawToFile(d,str(p))
        s=p.read_text();s=s.replace('font-family: BookSans','font-family: Arial Unicode MS, PingFang SC, sans-serif')
        s=s.replace('font-family="BookSans"','font-family="Arial Unicode MS, PingFang SC, sans-serif"')
        p.write_text(s)
    print(f'Generated {len(figures())} vector figures')
