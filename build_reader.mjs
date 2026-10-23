import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
let marked;
try { ({marked}=await import('marked')); }
catch (error) {
  if (!process.env.MARKED_MODULE) throw new Error('Install marked with npm install marked, or set MARKED_MODULE to its ESM entry path.', {cause:error});
  ({marked}=await import(process.env.MARKED_MODULE));
}

const root=path.dirname(fileURLToPath(import.meta.url));
const files=fs.readdirSync(path.join(root,'chapters')).filter(x=>x.endsWith('.md')).sort();
const chapters=files.map(file=>{
  let source=fs.readFileSync(path.join(root,'chapters',file),'utf8').replaceAll('../figures/','figures/');
  const n=Number(file.slice(0,2));
  const lab=path.join(root,'labs',`ch${String(n).padStart(2,'0')}.md`);
  if(n>=1&&n<=16&&fs.existsSync(lab)) source+='\n\n'+fs.readFileSync(lab,'utf8').replace(/^# /gm,'## ');
  source=source.replace(/[\u2010-\u2014]/g,'-');
  return {file,source,tokens:marked.lexer(source),title:source.split('\n')[0].replace(/^# /,'')};
});
fs.mkdirSync(path.join(root,'tmp'),{recursive:true});
fs.writeFileSync(path.join(root,'tmp','book-tokens.json'),JSON.stringify(chapters,null,2));
fs.writeFileSync(path.join(root,'book.md'),chapters.map(x=>x.source).join('\n\n---\n\n'));
const escape=s=>s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
const blocks=chapters.map((c,i)=>{
  let html=marked.parse(c.source);
  html=html.replace(/<img src="figures\/([^"/]+)\.svg" alt="([^"]*)">/g,(_,name,alt)=>{
    const p=path.join(root,'figures',name+'.svg');
    if(!fs.existsSync(p))throw new Error(`Missing ${p}`);
    let svg=fs.readFileSync(p,'utf8').replace(/<\?xml[^>]*>/,'').replace(/<!DOCTYPE[^>]*>/,'');
    svg=svg.replace(/<svg /,`<svg role="img" aria-label="${alt}" `);
    return `<figure>${svg}<figcaption>${alt}</figcaption></figure>`;
  });
  return `<section id="chapter-${i}" class="chapter">${html}</section>`;
}).join('\n');
const toc=chapters.map((c,i)=>`<a href="#chapter-${i}">${escape(c.title)}</a>`).join('');
const html=`<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>从张量到世界 · 技术书与TinyOmni实训</title>
<style>
:root{--ink:#172d40;--muted:#567083;--blue:#235b93;--paper:#fffefb;--line:#dce6ec}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#edf2f6;color:var(--ink);font:18px/1.9 -apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif}
nav{position:fixed;inset:0 auto 0 0;width:260px;padding:28px 20px;background:#142c40;color:white;overflow:auto;font-size:13px;line-height:1.55}nav b{display:block;color:#80d1d0;letter-spacing:2px;font-size:11px;margin-bottom:15px}nav h2{font-size:25px;font-weight:500;margin:0 0 7px}nav p{color:#b6cad8;font-size:12px;margin-bottom:22px}nav a{display:block;color:#dce7ef;text-decoration:none;padding:8px 2px;border-bottom:1px solid #2a4356}nav a:hover{color:#96e1d1}
main{margin-left:260px;padding:40px 32px 100px}.chapter{max-width:930px;margin:0 auto 36px;padding:54px 64px;background:var(--paper);box-shadow:0 4px 28px #16334e08;border:1px solid var(--line);border-radius:3px;scroll-margin-top:24px}h1{font-size:31px;line-height:1.5;font-weight:650;margin:0 0 40px;color:#173f61;border-bottom:3px solid #168384;padding-bottom:24px}h2{font-size:22px;line-height:1.5;margin:42px 0 18px;color:#185b78}h3{font-size:19px;margin-top:30px}p{margin:0 0 20px;text-align:justify}strong{color:#153d5a}a{color:#1a7485;text-underline-offset:3px}code{font:0.86em/1.6 'SFMono-Regular',Consolas,'PingFang SC',monospace;background:#eef3f6;border-radius:4px;padding:2px 5px}pre{background:#edf3f7;border-left:3px solid #41859e;padding:20px 22px;overflow:auto;line-height:1.65;font-size:14px;white-space:pre-wrap;overflow-wrap:anywhere}pre code{padding:0;background:none;font-size:inherit}table{width:100%;border-collapse:collapse;font-size:15px;line-height:1.7;margin:25px 0}th,td{padding:11px 13px;border:1px solid #d8e3eb;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#e9f0f6;color:#1b4c70}tr:nth-child(even){background:#f8fafb}figure{margin:27px -12px}figure svg{display:block;width:100%;height:auto}figcaption{font-size:13px;color:#627689;line-height:1.65;margin:10px 12px}blockquote{margin:24px 0;border-left:4px solid #d28b4f;padding:12px 20px;background:#fff4e9}ul,ol{padding-left:28px}li{margin:9px 0}.toolbar{position:fixed;right:18px;bottom:18px;z-index:10}button{border:1px solid #b3cad8;background:#fff;color:#224f6d;border-radius:8px;padding:10px 16px;font-size:13px;cursor:pointer}
@media(max-width:1000px){nav{width:210px}main{margin-left:210px;padding:24px 15px}.chapter{padding:36px 30px}body{font-size:17px}}@media(max-width:700px){nav{position:relative;width:auto;max-height:300px}nav a{display:inline-block;margin-right:16px}main{margin-left:0;padding:14px 8px}.chapter{padding:30px 20px}h1{font-size:27px}figure{margin:22px -10px}table{font-size:13px}}
@media print{nav,.toolbar{display:none}main{margin:0;padding:0}.chapter{box-shadow:none;border:0;margin:0;padding:0;page-break-before:always}body{background:white;font-size:11pt}h1{font-size:24pt}h2{font-size:16pt}figure,table,pre{break-inside:avoid}a{color:inherit}}
</style></head><body><nav><b>TENSORS TO WORLD</b><h2>从张量到世界</h2><p>首版书稿 · 2026.10.23<br>原理、矩阵图解与可运行训练项目<br>面向 RTX 3090 / 24 GB</p>${toc}</nav><main>${blocks}</main><div class="toolbar"><button onclick="window.print()">打印 / 保存</button></div></body></html>`;
fs.writeFileSync(path.join(root,'从张量到世界.html'),html);
console.log(JSON.stringify({chapters:chapters.length,characters:chapters.reduce((a,c)=>a+c.source.length,0),images:(html.match(/<figure>/g)||[]).length}));
