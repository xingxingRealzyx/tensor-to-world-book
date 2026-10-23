"""Eleven small train/save/load experiments supporting Chapters 01-11.

All data are synthetic. These experiments teach one mechanism at a time;
they are not reproductions of GPT-2, DeepSeek, or R1 training runs.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class Bigram(nn.Module):
    def __init__(self):
        super().__init__();self.table=nn.Embedding(8,8)
    def forward(self,x):return self.table(x)


class LookupAttention(nn.Module):
    def __init__(self):
        super().__init__();self.q=nn.Linear(4,16,bias=False)
        self.k=nn.Linear(4,16,bias=False);self.v=nn.Linear(2,16,bias=False)
        self.head=nn.Linear(16,2)
    def forward(self,x):
        memory,query=x
        a=(self.q(query)[:,None,:]@self.k(memory[:,:,:4]).transpose(1,2))/4
        a=a.softmax(-1)
        return self.head((a@self.v(memory[:,:,4:])).squeeze(1))


class SequenceRelation(nn.Module):
    def __init__(self):
        super().__init__();self.embed=nn.Embedding(2,32)
        self.pos=nn.Parameter(torch.randn(1,6,32)*.02)
        layer=nn.TransformerEncoderLayer(32,4,64,dropout=0,batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.head=nn.Linear(32,2)
    def forward(self,x):return self.head(self.enc(self.embed(x)+self.pos)[:,0])


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__();self.embed=nn.Embedding(8,32)
        self.pos=nn.Parameter(torch.randn(1,64,32)*.02)
        layer=nn.TransformerEncoderLayer(32,4,128,dropout=0,activation='gelu',batch_first=True,norm_first=True)
        self.blocks=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(32)
    def forward(self,x):
        t=x.shape[1]
        mask=torch.triu(torch.ones(t,t,device=x.device,dtype=torch.bool),diagonal=1)
        z=self.blocks(self.embed(x)+self.pos[:,:t],mask=mask)
        return self.norm(z)@self.embed.weight.T


class FFNComparison(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu=nn.Sequential(nn.Linear(4,24),nn.GELU(),nn.Linear(24,1))
        self.gate=nn.Linear(4,16);self.up=nn.Linear(4,16);self.down=nn.Linear(16,1)
    def forward(self,x):
        return self.gelu(x),self.down(F.silu(self.gate(x))*self.up(x))


class ToyMoE(nn.Module):
    def __init__(self):
        super().__init__();self.router=nn.Linear(2,4)
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(2,16),nn.Tanh(),nn.Linear(16,1)) for _ in range(4)])
    def forward(self,x):
        p=self.router(x).softmax(-1);w,idx=p.topk(2,dim=-1)
        w=w/w.sum(-1,keepdim=True)
        sparse=torch.zeros_like(p).scatter(1,idx,w)
        # For tiny teaching runs we compute every expert, then mask.
        # This is routing semantics, NOT an efficient sparse inference kernel.
        values=torch.stack([expert(x) for expert in self.experts],dim=1)
        y=(sparse[:,:,None]*values).sum(1)
        chosen=torch.zeros_like(p).scatter(1,idx,1.)
        load=chosen.mean(0)/2
        balance=4*(p.mean(0)*load.detach()).sum()
        return y,balance,load


class LowRankKV(nn.Module):
    def __init__(self):
        super().__init__();self.down=nn.Linear(8,3,bias=False)
        self.uk=nn.Linear(3,8,bias=False);self.uv=nn.Linear(3,6,bias=False)
    def forward(self,x):
        c=self.down(x);return self.uk(c),self.uv(c)


class RuleStudent(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.Embedding(10,16)
        self.net=nn.Sequential(nn.Flatten(),nn.Linear(32,96),nn.GELU(),nn.Linear(96,10))
    def forward(self,x):return self.net(self.emb(x))


def model_for(ch):
    return {1:lambda:nn.Linear(2,2),2:Bigram,3:LookupAttention,4:SequenceRelation,
            5:MiniGPT,6:MiniGPT,7:MiniGPT,8:FFNComparison,9:ToyMoE,
            10:LowRankKV,11:RuleStudent}[ch]()


def data(ch,batch,seed,device):
    g=torch.Generator().manual_seed(seed)
    randn=lambda *s:torch.randn(*s,generator=g)
    randint=lambda n,s:torch.randint(n,s,generator=g)
    if ch==1:
        x=randn(batch,2);y=x@torch.tensor([[2.,-1.],[.5,3.]])+torch.tensor([.2,-.3])
    elif ch in (2,5,6,7):
        # Independent sampled sequences; never cut overlapping windows across splits.
        starts=randint(8,(batch,1));whole=(starts+torch.arange(17)[None,:])%8
        x=whole[:,:-1];y=whole[:,1:]
    elif ch==3:
        order=torch.rand(batch,4,generator=g).argsort(-1)
        keys=F.one_hot(order,4).float();values=randint(2,(batch,4))
        memory=torch.cat((keys,F.one_hot(values,2).float()),-1)
        query_id=randint(4,(batch,));query=F.one_hot(query_id,4).float()
        location=(order==query_id[:,None]).long().argmax(-1)
        y=values.gather(1,location[:,None]).squeeze(1)
        x=(memory,query)
    elif ch==4:
        x=randint(2,(batch,6));y=(x[:,0]==x[:,-1]).long()
    elif ch==8:
        x=randn(batch,4);y=(torch.sin(x[:,0])+x[:,1]*x[:,2]-.3*x[:,3]).unsqueeze(-1)
    elif ch==9:
        x=randn(batch,2)
        y=torch.where(x[:,0]<0,torch.sin(3*x[:,0])+x[:,1],x[:,0]**2-2*x[:,1])[:,None]
    elif ch==10:
        x=randn(batch,8)
        teacher=torch.Generator().manual_seed(310)
        wd=torch.randn(8,3,generator=teacher)*.4
        uk=torch.randn(3,8,generator=teacher)*.4
        uv=torch.randn(3,6,generator=teacher)*.4
        y=(x@wd@uk,x@wd@uv)
    else:
        x=randint(10,(batch,2));y=x.sum(-1)%10
    def move(a):return tuple(move(b) for b in a) if isinstance(a,tuple) else a.to(device)
    return move(x),move(y)


def objective(ch,pred,target):
    if ch in (2,3,4,5,6,7,11):
        return F.cross_entropy(pred.reshape(-1,pred.shape[-1]),target.reshape(-1))
    if ch==8:return sum(F.mse_loss(p,target) for p in pred)
    if ch==9:return F.mse_loss(pred[0],target)+.01*pred[1]
    if ch==10:return sum(F.mse_loss(p,t) for p,t in zip(pred,target))
    return F.mse_loss(pred,target)


@torch.no_grad()
def evaluate(ch,model,device):
    model.eval();x,y=data(ch,512,900001,device);p=model(x)
    result={'loss':float(objective(ch,p,y))}
    if ch in (2,3,4,5,6,7,11):result['accuracy']=float((p.argmax(-1)==y).float().mean())
    if ch==8:result.update(gelu_mse=float(F.mse_loss(p[0],y)),swiglu_mse=float(F.mse_loss(p[1],y)))
    if ch==9:result.update(mse=float(F.mse_loss(p[0],y)),expert_load=p[2].tolist(),note='All experts computed before masking; no sparse speedup claimed.')
    if ch==10:
        c=model.down(x[:16]);k=model.uk(c);v=model.uv(c);q=x[:1]
        # nn.Linear stores [out,in], hence q @ uk.weight -> latent query.
        s_exp=q@k.T/math.sqrt(8);s_lat=(q@model.uk.weight)@c.T/math.sqrt(8)
        a=s_exp.softmax(-1);out_exp=a@v;out_lat=(a@c)@model.uv.weight.T
        result.update(score_equivalence_error=float((s_exp-s_lat).abs().max()),
                      output_equivalence_error=float((out_exp-out_lat).abs().max()),
                      note='Low-rank contents only; no RoPE or full DeepSeek MLA reproduction.')
    if ch==11:
        xx=torch.cartesian_prod(torch.arange(10),torch.arange(10)).to(device)
        result['all_100_in_domain_pairs_accuracy']=float((model(xx).argmax(-1)==xx.sum(-1)%10).float().mean())
        result['note']='Rule-teacher supervised distillation; not GRPO or reinforcement learning.'
    if ch in (1,8,9):
        pp=p[0] if isinstance(p,tuple) else p
        result.update(example_inputs=x[:3].tolist(),example_outputs=pp[:3].tolist(),example_targets=y[:3].tolist())
    if ch in (2,5,6,7):
        z=torch.tensor([[0]],device=device)
        for _ in range(15):z=torch.cat((z,model(z)[:,-1].argmax(-1,keepdim=True)),1)
        result['greedy_sequence']=z[0].tolist()
        if ch==7:
            torch.manual_seed(150)
            z=torch.tensor([[0]],device=device);temperature=.8;top_p=.9
            for _ in range(15):
                probs=(model(z)[:,-1]/temperature).softmax(-1)
                sorted_p,indices=probs.sort(descending=True)
                drop=(sorted_p.cumsum(-1)-sorted_p)>top_p
                sorted_p=sorted_p.masked_fill(drop,0);sorted_p/=sorted_p.sum(-1,keepdim=True)
                token=indices.gather(-1,torch.multinomial(sorted_p,1))
                z=torch.cat((z,token),-1)
            result['sampled_sequence']=z[0].tolist();result['sampling']={'temperature':temperature,'top_p':top_p}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chapter',type=int,required=True,choices=range(1,12))
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--load',type=Path)
    parser.add_argument('--predict',action='store_true')
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda','mps'])
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args();ch=args.chapter
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    model=model_for(ch).to(args.device)
    if args.load:
        saved=torch.load(args.load,map_location=args.device,weights_only=True)
        if saved['chapter']!=ch:raise ValueError('Checkpoint chapter does not match')
        model.load_state_dict(saved['state_dict'])
        result=evaluate(ch,model,args.device)
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    if args.predict:parser.error('--predict requires --load')
    if args.steps<1:parser.error('--steps must be positive')
    out=args.out or Path(f'runs/ch{ch:02d}');out.mkdir(parents=True,exist_ok=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003 if ch in (4,5,6,7) else .01,weight_decay=.001)
    before=evaluate(ch,model,args.device);history=[]
    initial=[p.detach().clone() for p in model.parameters()]
    for step in range(args.steps):
        # Even 32-bit seeds are reserved for training; validation uses odd 900001.
        # Do not rely on the high seed bit: some CPU RNGs ignore it.
        model.train();x,y=data(ch,args.batch_size,((args.seed*10000+step)*2) % (2**32),args.device)
        optimizer.zero_grad();loss=objective(ch,model(x),y)
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();history.append({'step':step+1,'loss':float(loss.detach())})
    after=evaluate(ch,model,args.device)
    changed=any(not torch.equal(a,b.detach()) for a,b in zip(initial,model.parameters()))
    assert changed,'No parameter changed'
    torch.save({'chapter':ch,'state_dict':model.state_dict(),'seed':args.seed,'steps':args.steps},out/'model.pt')
    restored=model_for(ch).to(args.device)
    restored.load_state_dict(torch.load(out/'model.pt',map_location=args.device,weights_only=True)['state_dict'])
    restored.eval();model.eval();x,y=data(ch,8,8000,args.device)
    def compare(a,b):
        if isinstance(a,tuple):
            for aa,bb in zip(a,b):compare(aa,bb)
        else:torch.testing.assert_close(a,b)
    compare(model(x),restored(x))
    report={'chapter':ch,'steps':args.steps,'device':args.device,
            'parameters':sum(p.numel() for p in model.parameters()),
            'before':before,'after':after,'parameter_update':changed,'reload_check':'PASS',
            'validation':'Fresh synthetic sampling seeds; not a held-out semantic-combination test.'}
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    with (out/'loss.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['step','loss']);writer.writeheader();writer.writerows(history)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
