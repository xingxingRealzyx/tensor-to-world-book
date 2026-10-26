/** Verify the actual inline engine using fixed examples and independent invariants.
 * Run from any working directory: node code/verify_tensor_lab.mjs
 * Does not train, download files, or modify artifacts.
 */
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const html=fs.readFileSync(new URL('../interactive/tensor-lab.html',import.meta.url),'utf8');
const match=html.match(/<script id="tensor-engine">([\s\S]*?)<\/script>/);
assert.ok(match,'The page must contain the inline tensor-engine');
const context=vm.createContext({});
vm.runInContext(match[1],context,{timeout:1000});
const E=context.TensorEngine;
const plain=x=>JSON.parse(JSON.stringify(x));
const close=(a,b,tol=1e-12)=>assert.ok(Math.abs(a-b)<=tol,`${a} != ${b}`);
let checks=0;
function test(name,fn){fn();checks++;console.log(`PASS ${name}`);}

test('matmul: all six outputs match independently calculated integers',()=>{
  assert.deepEqual(plain(E.matmulState().y),[[4,2],[1,2],[1,1]]);
  assert.deepEqual(plain(E.matmulState(0,0,5).terms.map(x=>x.product)),[1,0,2,1]);
  assert.deepEqual([0,1,2,3,4,5].map(step=>E.matmulState(0,0,step).partial),[0,1,1,3,4,4]);
  for(let r=0;r<3;r++)for(let c=0;c<2;c++)close(E.matmulState(r,c,5).partial,E.matmulState().y[r][c]);
});
test('multihead: fixed correct and incorrect layouts differ at the right addresses',()=>{
  const s=E.headState(4,3);
  assert.deepEqual(plain(s.heads),[[[[0,1],[4,5],[8,9]],[[2,3],[6,7],[10,11]]]]);
  assert.deepEqual(plain(s.wrong),[[[[0,1],[2,3],[4,5]],[[6,7],[8,9],[10,11]]]]);
  assert.deepEqual(plain(s.headIndex),[0,0,1,0]);
  assert.deepEqual(plain(s.wrongIndex),[0,0,2,0]);
  for(let value=0;value<12;value++){
    const t=E.headState(value),[b,h,n,j]=t.headIndex;
    assert.equal(t.heads[b][h][n][j],value);
    const [bs,ns,hs,js]=t.splitIndex;assert.equal(t.split[bs][ns][hs][js],value);
  }
  assert.deepEqual(Array.from(s.heads.flat(Infinity)).sort((a,b)=>a-b),Array.from({length:12},(_,i)=>i));
});
test('attention: exact dot products, analytical probabilities, weighted outputs',()=>{
  const s=E.attentionState(2,5),z=Math.exp(1/Math.sqrt(2));
  assert.deepEqual(plain(s.scores),[[1,0,1],[0,1,1],[1,1,2]]);
  const expected=[[1,0,0],[1/(1+z),z/(1+z),0],[1/(2+z),1/(2+z),z/(2+z)]];
  expected.forEach((row,i)=>row.forEach((v,j)=>close(s.weights[i][j],v)));
  close(s.output[0][0],1);close(s.output[0][1],0);
  close(s.output[1][0],1/(1+z));close(s.output[1][1],2*z/(1+z));
  close(s.output[2][0],(1+3*z)/(2+z));close(s.output[2][1],1);
  for(let i=0;i<3;i++){
    close(s.weights[i].reduce((a,b)=>a+b,0),1);
    for(let j=i+1;j<3;j++){assert.equal(s.masked[i][j],-Infinity);assert.equal(s.weights[i][j],0);}
  }
});
test('patches: fixed pixel order and every address round-trip',()=>{
  const s=E.patchState();
  assert.deepEqual(plain(s.patches),[[0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]]);
  assert.deepEqual(Array.from(s.patches.flat()).sort((a,b)=>a-b),Array.from({length:16},(_,i)=>i));
  for(let p=0;p<4;p++)for(let e=0;e<4;e++){
    const t=E.patchState(p,e,3);assert.equal(t.value,t.image[t.selected.y][t.selected.x]);
    assert.equal(2*Math.floor(t.selected.y/2)+Math.floor(t.selected.x/2),p);
    assert.equal(2*(t.selected.y%2)+t.selected.x%2,e);
  }
});
test('TinyOmni: actual slices, dimensions, frame order, and masks',()=>{
  const all=E.omniState();
  assert.deepEqual(plain(all.groups.map(g=>[g.kind,g.start,g.end])),[['cls',0,1],['text',1,6],['image',6,10],['audio',10,18],['video',18,34]]);
  assert.deepEqual(plain(all.sourceShape),[1,34,64]);assert.equal(all.tokens.length,34);assert.equal(all.visibleKeys.length,34);
  const absent=E.omniState({image:false,audio:false,video:false},[false,false,false,false,true],33,3);
  assert.equal(absent.tokens.length,34);assert.deepEqual(plain(absent.visibleKeys),[0,1,2,3,4]);
  assert.equal(absent.mask[0],false);assert.equal(absent.mask[5],true);assert.equal(absent.mask.slice(6).every(Boolean),true);
  assert.deepEqual(plain(absent.sourceShape),plain(absent.memoryShape));
  assert.equal(absent.selected.frame,3);assert.equal(absent.selected.patch,3);
  assert.deepEqual(plain(E.omniState(undefined,undefined,17).selected.samples),[224,256]);
  const audioOnly=E.omniState({image:false,audio:true,video:false});
  assert.deepEqual(plain(audioOnly.visibleKeys),[0,1,2,3,4,5,10,11,12,13,14,15,16,17]);
});
test('states are fresh and indices are validated',()=>{
  const a=E.matmulState();a.x[0][0]=99;assert.equal(E.matmulState().x[0][0],1);
  for(const fn of [()=>E.matmulState(3),()=>E.headState(12),()=>E.attentionState(-1),()=>E.patchState(4),()=>E.omniState(undefined,undefined,34)])assert.throws(fn);
});
test('all 24 UI steps render without missing values (DOM smoke, not layout QA)',()=>{
  // Minimal DOM sink exercises the real rendering code; browser QA checks layout.
  const element=(dataset={})=>({dataset,innerHTML:'',textContent:'',listeners:{},
    setAttribute(){},focus(){},addEventListener(type,fn){this.listeners[type]=fn;}});
  const ids=new Map(),nav=element(),tabs=Array.from({length:5},(_,i)=>element({scene:String(i)}));
  const document={getElementById(id){if(!ids.has(id))ids.set(id,element());return ids.get(id);},
    querySelectorAll(){return tabs;},querySelector(selector){return selector==='.scene-nav'?nav:element();}};
  const ui=vm.createContext({document});
  const scripts=[...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)];
  for(const script of scripts)vm.runInContext(script[1],ui,{timeout:1000});
  let rendered=0;
  for(let scene=0;scene<5;scene++){
    nav.listeners.click({target:{closest:()=>tabs[scene]}});
    for(let step=0;step<[6,4,6,4,4][scene];step++){
      const button=element({step:String(step)});
      ids.get('step-list').listeners.click({target:{closest:()=>button}});
      const output=ids.get('stage').innerHTML;
      assert.ok(output.length>500);assert.equal(/undefined|NaN/.test(output),false);
      rendered++;
    }
  }
  assert.equal(rendered,24);
  const toggle={dataset:{presence:'audio'},checked:false};
  ids.get('stage').listeners.change({target:toggle});
  assert.ok(ids.get('stage').innerHTML.includes('可读取 KEY：26'));
});
test('offline delivery has no external resources or network calls',()=>{
  assert.equal(/<(?:script|link|img)[^>]+(?:src|href)\s*=\s*["'](?:https?:)?\/\//i.test(html),false);
  assert.equal(/\b(?:fetch|XMLHttpRequest|WebSocket)\s*\(/.test(html),false);
  assert.ok(html.includes('prefers-reduced-motion:reduce'));
  assert.ok(html.includes('width=device-width'));
});
console.log(JSON.stringify({status:'PASS',checks,engine:'interactive/tensor-lab.html#tensor-engine',scenarios:5}));
