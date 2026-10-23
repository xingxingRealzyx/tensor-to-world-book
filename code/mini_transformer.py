"""A small, untrained NumPy decoder for tracing shapes and KV-cache equivalence.

Run: python3 mini_transformer.py
Requires NumPy. No weights are downloaded and no robot is controlled.
"""
import json
import numpy as np


def softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, eps=1e-5):
    # Fixed gamma=1, beta=0 in this teaching model.
    return (x-x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True)+eps)


def gelu(x):
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))


class TinyDecoder:
    def __init__(self, seed=7, vocab=16, width=8, heads=2, layers=2, context=32):
        if width % heads:
            raise ValueError('width must be divisible by heads')
        self.vocab, self.width, self.heads, self.context = vocab, width, heads, context
        rng=np.random.default_rng(seed)
        def w(shape): return rng.normal(0,0.1,shape)
        self.embedding=w((vocab,width));self.position=w((context,width))
        self.blocks=[{'qkv':w((width,3*width)), 'out':w((width,width)),
                      'up':w((width,4*width)), 'down':w((4*width,width))}
                     for _ in range(layers)]

    def forward(self, ids, cache=None):
        ids=np.asarray(ids,dtype=np.int64)
        if ids.ndim!=2 or ids.shape[1]==0:
            raise ValueError('ids must be non-empty [B,T]')
        B,T=ids.shape;D=self.width;H=self.heads;dh=D//H
        if np.any(ids<0) or np.any(ids>=self.vocab):
            raise ValueError('token ID outside vocabulary')
        if cache is not None and len(cache)!=len(self.blocks):
            raise ValueError('one cache entry is required per layer')
        past=0 if cache is None else cache[0][0].shape[2]
        if past+T>self.context:
            raise ValueError('position table exhausted')
        x=self.embedding[ids]+self.position[np.arange(past,past+T)][None,:,:]
        next_cache=[]
        for i,p in enumerate(self.blocks):
            norm=layer_norm(x)
            q,k,v=np.split(norm@p['qkv'],3,axis=-1)
            def heads(a): return a.reshape(B,T,H,dh).transpose(0,2,1,3)
            q,k,v=map(heads,(q,k,v))
            if cache is not None:
                old_k,old_v=cache[i]
                assert old_k.shape==(B,H,past,dh) and old_v.shape==old_k.shape
                k=np.concatenate((old_k,k),axis=2)
                v=np.concatenate((old_v,v),axis=2)
            # Absolute query positions matter for chunked cached decoding.
            allow=np.arange(past+T)[None,:] <= np.arange(past,past+T)[:,None]
            score=(q@k.swapaxes(-1,-2))/np.sqrt(dh)
            a=softmax(np.where(allow[None,None,:,:],score,-np.inf))
            out=(a@v).transpose(0,2,1,3).reshape(B,T,D)
            x=x+out@p['out']
            x=x+gelu(layer_norm(x)@p['up'])@p['down']
            next_cache.append((k.copy(),v.copy()))
        logits=layer_norm(x)@self.embedding.T
        return logits,next_cache


def self_check():
    model=TinyDecoder()
    ids=np.array([[1,2,3,4,5],[5,4,3,2,1]])
    full,_=model.forward(ids)
    cache=None;pieces=[]
    for t in range(ids.shape[1]):
        z,cache=model.forward(ids[:,t:t+1],cache)
        pieces.append(z)
    cached=np.concatenate(pieces,axis=1)
    np.testing.assert_allclose(full,cached,rtol=1e-10,atol=1e-10)
    # A multi-token prefill followed by a chunk must obey offset causal masking.
    pre,c=model.forward(ids[:,:3]);tail,_=model.forward(ids[:,3:],c)
    np.testing.assert_allclose(full,np.concatenate((pre,tail),axis=1),atol=1e-10)
    changed=ids.copy();changed[:,-1]=(changed[:,-1]+1)%model.vocab
    alternative,_=model.forward(changed)
    np.testing.assert_allclose(full[:,:-1],alternative[:,:-1],atol=1e-10)
    logits,_=model.forward(ids[:,:-1])
    probabilities=softmax(logits)
    targets=ids[:,1:]
    chosen=np.take_along_axis(probabilities,targets[...,None],axis=-1)[...,0]
    loss=float(-np.log(chosen).mean())
    return {'status':'PASS','logits_shape':list(full.shape),
            'cache_per_layer_shape':list(cache[0][0].shape),
            'max_full_vs_cached_error':float(np.abs(full-cached).max()),
            'random_model_next_token_loss':loss,
            'checks':['full versus one-token cached forward',
                      'prefill followed by chunked cached forward',
                      'future token cannot change earlier logits']}


if __name__=='__main__':
    print(json.dumps(self_check(),indent=2))
