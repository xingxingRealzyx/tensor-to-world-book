"""Build the vector-illustrated book PDF from the checked Markdown token stream."""
import json, re, html, copy, math
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
    Spacer, PageBreak, NextPageTemplate, Table, TableStyle, Flowable, KeepTogether)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.graphics import renderPDF
from build_figures import figures, BLUE, TEAL, INK, MUTED

ROOT=Path(__file__).parent
W,H=510.24,722.84
LEFT=46;RIGHT=44;TOP=48;BOTTOM=45
BW=W-LEFT-RIGHT;BH=H-TOP-BOTTOM
pdfmetrics.registerFont(TTFont('BookMono','/System/Library/Fonts/Menlo.ttc',subfontIndex=0))
pdfmetrics.registerFontFamily('BookSans',normal='BookSans',bold='BookSans',italic='BookSans',boldItalic='BookSans')

STYLES={
 'body':ParagraphStyle('body',fontName='BookSans',fontSize=10.4,leading=18.3,
    textColor=colors.HexColor(INK),wordWrap='CJK',spaceAfter=9,allowWidows=0,allowOrphans=0),
 'h1':ParagraphStyle('h1',fontName='BookSans',fontSize=24,leading=35,
    textColor=colors.HexColor(BLUE),wordWrap='CJK',spaceBefore=9,spaceAfter=24,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='BookSans',fontSize=14.5,leading=23,
    textColor=colors.HexColor(TEAL),wordWrap='CJK',spaceBefore=18,spaceAfter=10,keepWithNext=True),
 'h3':ParagraphStyle('h3',fontName='BookSans',fontSize=11.7,leading=20,
    textColor=colors.HexColor(BLUE),wordWrap='CJK',spaceBefore=12,spaceAfter=8,keepWithNext=True),
 'caption':ParagraphStyle('caption',fontName='BookSans',fontSize=8.3,leading=13,
    textColor=colors.HexColor(MUTED),wordWrap='CJK',spaceBefore=6,spaceAfter=13),
 'table':ParagraphStyle('table',fontName='BookSans',fontSize=8.7,leading=14,
    textColor=colors.HexColor(INK),wordWrap='CJK',spaceAfter=0),
 'quote':ParagraphStyle('quote',fontName='BookSans',fontSize=10,leading=17,
    textColor=colors.HexColor(MUTED),wordWrap='CJK',leftIndent=12,spaceAfter=10),
}

def esc(s):return html.escape(str(s),quote=True)

def inline(tokens):
    ans=[]
    for t in tokens or []:
        typ=t['type']
        if typ=='text':ans.append(inline(t['tokens']) if 'tokens'in t else esc(t.get('text','')))
        elif typ=='codespan':ans.append(f'<font color="#285b80">{esc(t["text"])}</font>')
        elif typ=='strong':ans.append(f'<font color="#103d5c">{inline(t.get("tokens",[]))}</font>')
        elif typ in ('em','del'):ans.append(inline(t.get('tokens',[])))
        elif typ=='link':ans.append(f'<link href="{esc(t["href"])}" color="#167b84">{inline(t.get("tokens",[]))}</link>')
        elif typ=='br':ans.append('<br/>')
        elif typ=='escape':ans.append(esc(t.get('text','')))
        elif typ=='image':continue
        elif typ=='html':ans.append('<br/>' if '<br' in t.get('text','') else '')
        else:ans.append(esc(t.get('text','')))
    return ''.join(ans)

class VectorFigure(Flowable):
    def __init__(self,drawing):
        super().__init__();self.drawing=drawing
        self.width=BW;self.height=drawing.height*(BW/drawing.width)
        self.keepWithNext=True
    def draw(self):
        self.canv.saveState();self.canv.scale(BW/self.drawing.width,BW/self.drawing.width)
        renderPDF.draw(self.drawing,self.canv,0,0);self.canv.restoreState()

def mono_font(char):return 'BookMono' if ord(char)<128 else 'BookSans'
def textwidth(s,size):return sum(pdfmetrics.stringWidth(c,mono_font(c),size) for c in s)

class CodeBlock(Flowable):
    def __init__(self,text=None,lines=None,language='text'):
        super().__init__();self.width=BW;self.fs=8.2;self.lh=12.8;self.language=language
        self.lines=[]
        if lines is not None:self.lines=lines
        else:
            for line in text.expandtabs(4).splitlines() or ['']:
                while textwidth(line,self.fs)>BW-24:
                    n=1
                    while n<len(line) and textwidth(line[:n+1],self.fs)<=BW-28:n+=1
                    split_at=line.rfind(' ',0,n)
                    if split_at>8:n=split_at
                    suffix=' \\' if language in ('bash','sh','shell') else ''
                    self.lines.append(line[:n]+suffix);line='  '+line[n:].lstrip()
                self.lines.append(line)
        self.height=len(self.lines)*self.lh+20
    def wrap(self,aw,ah):return self.width,self.height
    def split(self,aw,ah):
        n=int((ah-20)/self.lh)
        if n<2 or n>=len(self.lines):return []
        return [CodeBlock(lines=self.lines[:n],language=self.language),CodeBlock(lines=self.lines[n:],language=self.language)]
    def draw(self):
        c=self.canv;c.saveState();c.setFillColor(colors.HexColor('#EDF3F7'))
        c.roundRect(0,0,self.width,self.height,5,fill=1,stroke=0)
        c.setStrokeColor(colors.HexColor('#568EA7'));c.setLineWidth(2);c.line(0,2,0,self.height-2)
        c.setFillColor(colors.HexColor('#264257'))
        for i,line in enumerate(self.lines):
            x=11;y=self.height-14-i*self.lh
            runs=[]
            for char in line:
                font=mono_font(char)
                if runs and runs[-1][0]==font:runs[-1][1]+=char
                else:runs.append([font,char])
            for font,txt in runs:
                c.setFont(font,self.fs);c.drawString(x,y,txt);x+=pdfmetrics.stringWidth(txt,font,self.fs)
        c.restoreState()

class BookDoc(BaseDocTemplate):
    def __init__(self,path):
        super().__init__(str(path),pagesize=(W,H),leftMargin=LEFT,rightMargin=RIGHT,
                         topMargin=TOP,bottomMargin=BOTTOM,
                         title='从张量到世界：Transformer、语言模型与多模态智能',
                         author='为学习与实践编写的中文技术书稿',allowSplitting=True)
        self.chapter_title='从张量到世界';self.head_index=0;self.locations=[]
        frame=Frame(LEFT,BOTTOM,BW,BH,id='body',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        self.addPageTemplates([PageTemplate(id='Cover',frames=[frame],onPage=cover),
                               PageTemplate(id='Body',frames=[frame],onPageEnd=running)])
    def beforeDocument(self):self.head_index=0;self.locations=[];self.chapter_title='从张量到世界'
    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and hasattr(flowable,'book_heading'):
            level,title=flowable.book_heading
            key=f'heading-{self.head_index}';self.head_index+=1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title,key,level=level,closed=level>0)
            self.locations.append({'level':level,'title':title,'page':self.page})
            if level==0:
                self.chapter_title=title
                self.notify('TOCEntry',(0,title,self.page,key))

def cover(c,doc):
    c.saveState();c.setFillColor(colors.HexColor('#112C41'));c.rect(0,0,W,H,fill=1,stroke=0)
    c.setFillColor(colors.HexColor('#8DD4D0'));c.setFont('Helvetica',10)
    c.drawString(48,H-65,'TENSORS TO WORLD  /  A VISUAL TECHNICAL BOOK')
    c.setFont('BookSans',35);c.setFillColor(colors.white);c.drawString(45,H-139,'从张量到世界')
    c.setFont('BookSans',16);c.setFillColor(colors.HexColor('#C4D9E6'))
    c.drawString(48,H-181,'Transformer、语言模型与多模态智能')
    c.setFont('BookSans',12);c.drawString(48,H-216,'用矩阵理解模型，用训练验证理解')
    ox,oy=53,240
    for r in range(5):
        for col in range(9):
            if col<=r+3:
                palette=['#27608B','#2F8198','#45A8A0','#85C9B7']
                c.setFillColor(colors.HexColor(palette[(r+col)%4]));c.roundRect(ox+col*43,oy+r*30,35,23,3,fill=1,stroke=0)
    c.setStrokeColor(colors.HexColor('#72C8C0'));c.line(48,205,W-48,205)
    c.setFont('BookSans',11);c.setFillColor(colors.HexColor('#E3EEF4'))
    c.drawString(48,176,'GPT-2  ·  DeepSeek  ·  VLM  ·  GPT-4o  ·  VLA')
    c.drawString(48,150,'逐章实训与 TinyOmni：文字、图像、声音的最小闭环')
    c.setFont('BookSans',10);c.setFillColor(colors.HexColor('#A8C3D3'))
    c.drawString(48,92,'纸面分镜图解版  /  2026年10月26日')
    c.drawString(48,72,'训练配置面向 RTX 3090 24 GB；随书提供源码与验证记录')
    c.restoreState()

def running(c,doc):
    c.saveState();c.setFillColor(colors.HexColor(MUTED));c.setFont('BookSans',8)
    c.drawString(LEFT,H-27,'从张量到世界')
    label=doc.chapter_title
    while pdfmetrics.stringWidth(label,'BookSans',8)>BW-100:label=label[:-2]+'…' if not label.endswith('…') else label[:-2]+'…'
    c.drawRightString(W-RIGHT,H-27,label)
    c.setStrokeColor(colors.HexColor('#D6E1E8'));c.setLineWidth(.5);c.line(LEFT,H-34,W-RIGHT,H-34)
    c.setFont('BookSans',8);c.drawString(LEFT,24,'纸面分镜图解版 · 原理与实训')
    c.setFont('Helvetica',9);c.drawRightString(W-RIGHT,24,str(doc.page))
    c.restoreState()

FIGURES=figures()

def leading_space(height):
    spacer=Spacer(1,height)
    spacer.keepWithNext=True
    return spacer

def render_blocks(tokens,chapter_start=False):
    out=[]
    for token in tokens:
        typ=token['type']
        if typ=='space':continue
        if typ=='heading':
            depth=token['depth'];txt=inline(token.get('tokens',[]))
            p=Paragraph(txt,STYLES['h1' if depth==1 else 'h2' if depth==2 else 'h3'])
            p.book_heading=(0 if depth==1 else 1,token['text'])
            out.append(p)
        elif typ in ('paragraph','text'):
            inlines=token.get('tokens',[])
            images=[t for t in inlines if t.get('type')=='image']
            if images:
                for im in images:
                    name=Path(im['href']).stem
                    if name not in FIGURES:raise RuntimeError(f'Missing illustration: {name}')
                    cap=Paragraph(esc(im.get('text','')),STYLES['caption'])
                    out.extend([leading_space(6),VectorFigure(FIGURES[name]),cap])
            else:
                text=inline(inlines) if inlines else esc(token.get('text',''))
                if text.strip():out.append(Paragraph(text,STYLES['body']))
        elif typ=='code':out.extend([CodeBlock(token['text'],language=token.get('lang','text')),Spacer(1,10)])
        elif typ=='table':
            cells=[token['header']]+token['rows']
            data=[[Paragraph(inline(c.get('tokens',[])) or esc(c.get('text','')),STYLES['table']) for c in row] for row in cells]
            n=len(cells[0]);widths=[BW/n]*n
            if n==3:widths=[BW*.25,BW*.36,BW*.39]
            table=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E6EFF5')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F7FAFC')]),
                ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D6E2EA')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
                ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),
                ('BOTTOMPADDING',(0,0),(-1,-1),7)]))
            out.extend([leading_space(3),table,Spacer(1,11)])
        elif typ=='list':
            for i,item in enumerate(token['items']):
                prefix=f'{i+token.get("start",1)}. ' if token.get('ordered') else '• '
                first=True
                for child in item.get('tokens',[]):
                    if first and child['type'] in ('text','paragraph'):
                        txt=inline(child.get('tokens',[])) or esc(child.get('text',''))
                        out.append(Paragraph(esc(prefix)+txt,STYLES['body']));first=False
                    else:out.extend(render_blocks([child]))
        elif typ=='blockquote':out.extend(render_blocks(token['tokens']))
        elif typ=='hr':out.append(Spacer(1,15))
    return out

def build():
    chapters=json.loads((ROOT/'tmp/book-tokens.json').read_text())
    target=ROOT/'output/pdf/从张量到世界.pdf';target.parent.mkdir(parents=True,exist_ok=True)
    story=[Spacer(1,BH-5),NextPageTemplate('Body'),PageBreak()]
    story.append(Paragraph('目录',STYLES['h1']))
    story.append(Paragraph('按章节循序阅读；每章实训与第17章项目共同构成可运行的学习路线。',STYLES['body']))
    toc=TableOfContents();toc.levelStyles=[ParagraphStyle('toc',fontName='BookSans',fontSize=11,
        leading=21,textColor=colors.HexColor(BLUE),wordWrap='CJK',spaceBefore=6,leftIndent=0,rightIndent=20)]
    story.extend([toc,PageBreak()])
    for i,ch in enumerate(chapters):
        if i:story.append(PageBreak())
        story.extend(render_blocks(ch['tokens']))
    doc=BookDoc(target);doc.multiBuild(story,maxPasses=4)
    (ROOT/'tmp/page-map.json').write_text(json.dumps(doc.locations,ensure_ascii=False,indent=2))
    print(json.dumps({'pdf':str(target),'pages':doc.page,'headings':len(doc.locations)},ensure_ascii=False))

if __name__=='__main__':build()
