"""Tensor-first illustrations: every grid cell is a value or an explicit index."""
import math

INK='#172D40';BLUE='#235B93';TEAL='#087F82';ORANGE='#CC702D';MUTED='#536D7E'
ROW=['#D9EAF6','#DCF0E6','#F9E5CE'];COL=['#E4DDF5','#DDF0EB','#F9E5CE','#DFEAF7']

def grid(f,x,y,data,cell=40,rows=None,cols=None,selected=None):
    for r,row in enumerate(data):
        for c,v in enumerate(row):
            fill=rows[r%len(rows)] if rows else cols[c%len(cols)] if cols else '#E6EEF5'
            picked=selected is not None and (r,c) in selected
            f.rect(x+c*cell,y+r*cell,cell-2,cell-2,'#FFE0A8' if picked else fill,
                   stroke=ORANGE if picked else None,radius=2)
            f.txt(x+(c+.5)*cell-1,y+(r+.63)*cell,str(v),min(16,cell*.37),INK,'middle')

def build(Figure):
    out={}
    f=Figure('张量块：把批次轴摊开来看','格子是数；行是位置T；列是通道D；每一片是样本B',540)
    a=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    b=[[13,14,15,16],[17,18,19,20],[21,22,23,24]]
    f.txt(52,101,'X[0]：第0个样本',18,BLUE);grid(f,52,120,a,44,rows=ROW)
    f.txt(380,101,'X[1]：第1个样本',18,TEAL);grid(f,380,120,b,44,rows=ROW,selected={(2,3)})
    for x in (52,380):
        f.arrow(x,274,x+174,274,BLUE);f.txt(x+78,298,'D：向右找列',15,BLUE,'middle')
        f.arrow(x-17,125,x-17,250,TEAL)
    f.tensor(60,374,3,4,depth=2,cell=27,values=[a,b],label='叠起来：[B=2,T=3,D=4]')
    f.lines(345,367,['X[1,2,3] = 24','先找第1片，再找第2行，第3列。','一共 2 × 3 × 4 = 24 个数。','立体叠片表示索引轴，不表示物理空间。'],17,INK,35)
    f.txt(28,521,'本书索引从0开始；读数字时摊开切片，读形状时看张量块。',15,MUTED)
    out['01-tensor-cells']=f.d

    f=Figure('矩阵乘法：一个输出格怎样算出来','X的第0行 × W的第0列；沿共同轴D逐项相乘，再相加',520)
    X=[[1,0,2,-1],[0,1,1,0],[2,1,0,1]];W=[[1,0],[0,1],[1,1],[-1,0]]
    Y=[[4,2],[1,2],[1,1]]
    f.txt(30,101,'X [T=3,D=4]',17,BLUE);grid(f,30,125,X,40,selected={(0,c) for c in range(4)})
    f.txt(245,101,'W [D=4,F=2]',17,TEAL);grid(f,245,125,W,40,selected={(r,0) for r in range(4)})
    f.txt(479,101,'Y [T=3,F=2]',17,ORANGE);grid(f,479,125,Y,48,selected={(0,0)})
    f.txt(210,196,'@',24,INK,'middle');f.arrow(348,190,449,190)
    f.lines(32,329,['输出 Y[0,0] 只读橙色的一行与一列：','1×1   +   0×0   +   2×1   +   (-1)×(-1)  =  4'],18,INK,36)
    f.line(32,389,668,389,'#C8D8E3')
    f.lines(32,421,['换一个输出格，就换一行、一列；W仍是同一份。','T留下来，F留下来，共同的D在求和中消失。','这一步混合通道，不会把第0行与第1行相加。'],17,INK,32)
    out['01-tensor-matmul']=f.d

    f=Figure('广播：同一条偏置加到每一行','Y [3,2] + b [2] → Z [3,2]；不是每个位置学一条不同的b',440)
    f.txt(30,99,'Y',18,BLUE);grid(f,30,123,[[4,2],[1,2],[1,1]],48,rows=ROW)
    f.txt(230,99,'b = [10,-1]',18,TEAL);grid(f,230,123,[[10,-1]],48,cols=COL)
    f.txt(490,99,'Z',18,ORANGE);grid(f,490,123,[[14,1],[11,1],[11,0]],48,rows=ROW)
    for r in range(3):f.arrow(333,147,476,147+r*48,TEAL,dash=[3,3])
    f.lines(30,322,['第0行：[4,2] + [10,-1] = [14,1]','第1行：[1,2] + [10,-1] = [11,1]','虚线表示重复使用同一参数；不会增加三份偏置参数。'],18,INK,34)
    out['01-tensor-broadcast']=f.d

    f=Figure('reshape与transpose：同形状，不同格子','从 [2,3] 出发，两种操作都能得到 [3,2]，但索引关系不同',520)
    f.txt(40,100,'原数组 X [2,3]',18,BLUE);grid(f,40,123,[[1,2,3],[4,5,6]],45,rows=ROW)
    f.txt(365,100,'线性次序：[1,2,3,4,5,6]',17,MUTED)
    f.txt(47,276,'reshape(3,2)',18,TEAL);grid(f,47,296,[[1,2],[3,4],[5,6]],45)
    f.txt(371,276,'transpose(1,0)',18,ORANGE);grid(f,371,296,[[1,4],[2,5],[3,6]],45,cols=ROW)
    f.arrow(110,222,110,257,TEAL);f.arrow(211,162,411,244,ORANGE)
    f.lines(47,467,['按原线性顺序重新分组','新[0,1] = 2'],16,TEAL,25)
    f.lines(371,467,['交换“行号”和“列号”','新[0,1] = 原[1,0] = 4'],16,ORANGE,25)
    out['01-tensor-reshape']=f.d

    f=Figure('沿哪根轴求和，决定留下什么','同一张表；沿列收拢得到每行的和，沿行收拢得到每列的和',420)
    f.txt(35,99,'X [T=2,D=3]',18,BLUE);grid(f,35,124,[[1,2,3],[4,5,6]],52,rows=ROW)
    f.arrow(214,174,325,174,TEAL);grid(f,350,124,[[6],[15]],52,rows=ROW)
    f.lines(429,147,['sum(axis=1)','[2,1] 保留T'],17,TEAL,29)
    f.arrow(111,240,111,287,ORANGE);grid(f,35,307,[[5,7,9]],52,cols=COL)
    f.lines(261,320,['sum(axis=0) → [1,3] 保留D','keepdims=True保留长度1的轴，方便广播回去。'],17,INK,31)
    out['01-tensor-reduce']=f.d

    f=Figure('Embedding：按编号整行取出，不做乘法','ids=[2,0,2]；重复编号会先取到同一行，再由上下文改变',495)
    E=[[1,0,0],[0,2,0],[1,1,3],[2,-1,0]]
    f.txt(42,100,'E [V=4,D=3]',18,BLUE);grid(f,62,125,E,44,rows=['#D9EAF6','#E8EEF2','#F9E5CE','#DCF0E6'])
    for r in range(4):f.txt(40,153+r*44,str(r),16,MUTED,'middle')
    f.txt(405,100,'X [T=3,D=3]',18,TEAL);grid(f,405,125,[E[2],E[0],E[2]],44,rows=['#F9E5CE','#D9EAF6','#F9E5CE'])
    for y1,y2,lab in [(235,147,'2'),(147,191,'0'),(235,235,'2')]:
        f.arrow(203,y1,393,y2,ORANGE if lab=='2' else BLUE)
    f.lines(40,353,['第0个位置：ids[0]=2，所以 X[0,:]=E[2,:]。','第2个位置也查第2行；这里不会自动区分两次出现。','词表轴V被“查地址”消费掉，位置轴T由输入编号的个数决定。','可训练参数只有E；ids是整数地址，不是可学习向量。'],17,INK,34)
    out['02-tensor-gather']=f.d

    f=Figure('注意力第一乘：每个查询对每个键打分','先把K转置，让一个查询行能依次与每个键列做点积',500)
    q=[[1,0],[0,1],[1,1]];kt=[[1,0,1],[0,1,1]];s=[[1,0,1],[0,1,1],[1,1,2]]
    f.txt(30,100,'Q [Tq=3,d=2]',17,BLUE);grid(f,30,125,q,42,selected={(2,0),(2,1)})
    f.txt(232,100,'K.T [d=2,Tk=3]',17,TEAL);grid(f,232,125,kt,42,selected={(0,2),(1,2)})
    f.txt(487,100,'Q @ K.T [3,3]',17,ORANGE);grid(f,487,125,s,48,selected={(2,2)})
    f.txt(166,192,'@',25,INK);f.arrow(373,185,464,185)
    f.lines(30,324,['橙色格 (Q@K.T)[2,2] = [1,1] · [1,1] = 2。','输出的行号来自查询位置Tq，列号来自键位置Tk。','比较用的d轴被求和消去；这张表不再是“位置×特征”。','在下一张图里，沿每行的Tk轴把分数变成读取比例。'],18,INK,36)
    out['03-tensor-score']=f.d

    f=Figure('分数变权重：缩放、遮挡、逐行归一化','因果掩码把未来键置为-inf；exp(-inf)=0',555)
    f.txt(30,99,'先除以√2，再加掩码',18,BLUE)
    grid(f,30,123,[[.707,'-inf','-inf'],[0,.707,'-inf'],[.707,.707,1.414]],62,rows=ROW)
    f.txt(399,99,'A [Tq=3,Tk=3]',18,TEAL)
    grid(f,399,123,[[1,0,0],[.330,.670,0],[.248,.248,.503]],62,rows=ROW)
    f.arrow(237,215,377,215,TEAL);f.txt(302,191,'逐行softmax',15,TEAL,'middle')
    f.lines(30,359,['只展开索引1的行：分数 [0,0.7071,-inf]','减最大值 → [-0.7071,0,-inf]','取指数 → [0.4931,1,0]；总和约1.4931','除以总和 → [0.3302,0.6698,0]','每行独立归一化；显示值四舍五入，计算使用完整精度。'],17,INK,35)
    out['03-tensor-softmax']=f.d

    f=Figure('注意力第二乘：按比例混合整条值向量','固定第2个查询：A[2,:]的三个权重，分别缩放V的三行',570)
    f.txt(34,101,'V [Tk=3,dv=2]',18,BLUE);grid(f,34,124,[[1,0],[0,2],[3,1]],79,rows=ROW)
    f.txt(215,101,'逐行乘对应权重',18,TEAL)
    for r,lab in enumerate(['× 0.248255…','× 0.248255…','× 0.503490…']):
        f.txt(216,174+r*79,lab,17,INK);f.arrow(360,164+r*79,394,164+r*79,TEAL)
    grid(f,408,124,[[.2483,0],[0,.4965],[1.5105,.5035]],79,rows=ROW)
    f.txt(408,384,'沿Tk加起来，保留dv列',17,ORANGE)
    f.arrow(482,394,482,413,ORANGE);grid(f,408,425,[[1.7587,1.0000]],79,cols=COL)
    f.lines(31,402,['第一列：','0.2483+0+1.5105','第二列：','0+0.4965+0.5035'],17,INK,30)
    f.txt(30,550,'结果是一条新向量 O[2,:]，不是直接输出一个词。',17,INK)
    out['03-tensor-mix']=f.d

    f=Figure('多头：切的是通道，不是前后半句','B=1,T=3,D=4,H=2；数字是元素编号，颜色标记原token',620)
    f.txt(35,100,'X[0] [T=3,D=4]',18,BLUE);grid(f,35,124,[[0,1,2,3],[4,5,6,7],[8,9,10,11]],40,rows=ROW)
    f.line(113,119,113,246,ORANGE,width=3)
    f.txt(290,100,'正确：每个头都有3个位置',18,TEAL)
    grid(f,290,124,[[0,1],[4,5],[8,9]],40,rows=ROW)
    grid(f,484,124,[[2,3],[6,7],[10,11]],40,rows=ROW)
    f.txt(329,273,'head 0',16,TEAL,'middle');f.txt(523,273,'head 1',16,TEAL,'middle')
    f.arrow(201,164,278,164,TEAL)
    f.txt(35,328,'reshape [1,3,2,2] → transpose(1,2) → [1,2,3,2]',17,INK)
    f.txt(35,382,'错误：直接 reshape [1,2,3,2]',18,ORANGE)
    grid(f,35,404,[[0,1],[2,3],[4,5]],40);grid(f,240,404,[[6,7],[8,9],[10,11]],40)
    f.lines(385,425,['错误head 0的第1行是[2,3]，','它仍来自原token 0。','正确位置应为token 1的[4,5]。'],16,INK,32)
    f.txt(35,582,'合并时先把H移回T后面，再把H与dh合成D。',18,TEAL)
    out['03-tensor-heads']=f.d

    f=Figure('LayerNorm：每个位置自己整理通道','先只看第0行；不会拿另一句、另一个token的数来求均值',530)
    f.txt(35,100,'X [T=2,D=4]',18,BLUE);grid(f,35,123,[[1,3,5,7],[2,4,6,8]],47,rows=ROW)
    f.txt(343,100,'第0行的均值 = 4',18,TEAL)
    grid(f,343,123,[[-3,-1,1,3]],64,rows=ROW)
    f.txt(343,218,'每格减4，得到上面一行。',17,INK)
    f.lines(35,291,['平方后：[9,1,1,9]，均值 = 5。','每格再除以 sqrt(5+ε)，这里 ε=0.00001。','γ=1、β=0时，输出近似为：'],18,INK,35)
    grid(f,35,396,[[-1.342,-.447,.447,1.342]],86,rows=ROW)
    f.lines(414,415,['四格合起来才是','一个token的状态。'],17,TEAL,31)
    f.txt(35,514,'真实模型还学习逐通道γ和β；这两组参数在各位置共享。',16,MUTED)
    out['04-tensor-norm']=f.d

    f=Figure('残差相加：对应格相加，轴完全对齐','同一位置、同一通道配对；下例只演示相加，不是整层实测',440)
    f.txt(35,102,'主路径 X [2,2]',18,BLUE);grid(f,35,128,[[1,2],[3,4]],56,rows=ROW)
    f.txt(255,102,'增量 Δ [2,2]',18,TEAL);grid(f,255,128,[[.1,-.2],[.3,.4]],56,rows=ROW)
    f.txt(478,102,'U = X+Δ',18,ORANGE);grid(f,478,128,[[1.1,1.8],[3.3,4.4]],56,rows=ROW)
    f.txt(190,196,'+',26,INK);f.txt(411,196,'=',26,INK)
    f.lines(35,311,['U[0,1] = 2 + (-0.2) = 1.8。','分支输出必须回到主路径的D维，才能逐格相加。','“保留形状”不等于“保留数值”：每个格都可以被更新。'],18,INK,36)
    out['04-tensor-residual']=f.d

    f=Figure('前馈网络：放宽通道，再收回原宽度','逐token重复同一计算；本图只展开一行，不混合位置',535)
    f.txt(30,99,'x [1,2]',18,BLUE);grid(f,30,124,[[1,-1]],44,rows=ROW)
    f.txt(236,99,'Wup [2,4]',18,TEAL);grid(f,236,124,[[1,0,1,1],[0,1,1,-1]],44)
    f.txt(140,158,'@',24,INK);f.arrow(428,163,474,163)
    f.txt(499,99,'隐层 [1,4]',18,ORANGE);grid(f,499,124,[[1,-1,0,2]],40)
    f.txt(30,274,'ReLU逐格：负数变0',18,TEAL);grid(f,30,299,[[1,0,0,2]],44)
    f.txt(279,274,'Wdown [4,2]',18,TEAL);grid(f,279,299,[[1,0],[0,1],[1,1],[1,-1]],40)
    f.txt(500,274,'输出 [1,2]',18,ORANGE);grid(f,500,299,[[3,-2]],56)
    f.txt(230,354,'@',24,INK);f.arrow(377,350,480,350)
    f.lines(30,488,['输出第0格 = 1×1+0×0+0×1+2×1 = 3。','ReLU只为手算方便；GPT-2用GELU，现代模型还常用门控。'],16,INK,29)
    out['04-tensor-ffn']=f.d
    return out
