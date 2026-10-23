"""Verify the numerical examples used in the book, without training a model."""
import json
from pathlib import Path
import numpy as np
from mini_transformer import softmax, self_check


def verify():
    q=np.array([[1,0],[0,1],[1,1]],dtype=float)
    v=np.array([[1,0],[0,2],[3,1]],dtype=float)
    a=softmax(np.where(np.tril(np.ones((3,3),dtype=bool)),q@q.T/np.sqrt(2),-np.inf))
    out=a@v
    np.testing.assert_allclose(a.sum(-1),1)
    assert np.all(a[np.triu_indices(3,k=1)]==0)
    D=768;V=50257;L=12;ctx=1024
    block=4*D*D+4*D+8*D*D+5*D+4*D
    total=V*D+ctx*D+L*block+2*D
    assert block==7087872 and total==124439808
    mha=1*8192*32*32*(128+128)*2
    gqa=mha//4
    assert mha==4*2**30 and gqa==2**30
    mla=1*32768*61*(512+64)*2
    expanded=1*32768*61*(128*(128+128)+64)*2
    c=np.array([[1.,0],[0,2]])
    uk=np.array([[1.,0],[1,1]])
    uv=np.array([[2.,0],[0,1]])
    query=np.array([[1.,2]])
    np.testing.assert_allclose(query@(c@uk).T,(query@uk.T)@c.T)
    weights=softmax(query@(c@uk).T/np.sqrt(2))
    np.testing.assert_allclose(weights@(c@uv),(weights@c)@uv)
    rewards=np.array([1.,0,0,0]);adv=(rewards-rewards.mean())/rewards.std(ddof=0)
    np.testing.assert_allclose(adv,[np.sqrt(3),-1/np.sqrt(3),-1/np.sqrt(3),-1/np.sqrt(3)])
    clip=softmax(np.array([[3,1,0],[0,3,1],[1,0,3]],dtype=float))
    def rot(a):return np.array([np.cos(np.deg2rad(a)),np.sin(np.deg2rad(a))])
    np.testing.assert_allclose(rot(60)@rot(180),-.5,atol=1e-15)
    np.testing.assert_allclose(rot(120)@rot(240),-.5,atol=1e-15)
    return {'status':'PASS','attention_weights':a.tolist(),'attention_output':out.tolist(),
            'gpt2_small_parameters':total,'gpt2_block_parameters':block,
            'mha_cache_GiB':mha/2**30,'gqa_cache_GiB':gqa/2**30,
            'mla_cache_GiB':mla/2**30,'expanded_mla_cache_GiB':expanded/2**30,
            'mla_example_weights':weights.tolist(),'grpo_advantage':adv.tolist(),
            'clip_row_probabilities':clip.tolist(),'tiny_decoder':self_check()}


if __name__=='__main__':
    result=verify()
    path=Path(__file__).with_name('verification.json')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
