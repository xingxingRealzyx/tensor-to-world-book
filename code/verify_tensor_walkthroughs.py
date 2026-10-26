"""Check the exact numbers and index mappings used in the tensor diagrams."""
import json
from pathlib import Path
import numpy as np

def verify():
    x=np.array([[1,0,2,-1],[0,1,1,0],[2,1,0,1]])
    w=np.array([[1,0],[0,1],[1,1],[-1,0]])
    y=x@w
    np.testing.assert_array_equal(y,[[4,2],[1,2],[1,1]])
    products=x[0]*w[:,0]
    np.testing.assert_array_equal(products,[1,0,2,1])
    np.testing.assert_array_equal(np.cumsum(products),[1,1,3,4])
    np.testing.assert_array_equal(y+[10,-1],[[14,1],[11,1],[11,0]])
    a=np.arange(1,7).reshape(2,3)
    np.testing.assert_array_equal(a.reshape(3,2),[[1,2],[3,4],[5,6]])
    np.testing.assert_array_equal(a.T,[[1,4],[2,5],[3,6]])
    np.testing.assert_array_equal(a.sum(1,keepdims=True),[[6],[15]])
    np.testing.assert_array_equal(a.sum(0,keepdims=True),[[5,7,9]])
    e=np.array([[1,0,0],[0,2,0],[1,1,3],[2,-1,0]])
    np.testing.assert_array_equal(e[[2,0,2]],[[1,1,3],[1,0,0],[1,1,3]])
    q=np.array([[1.,0.],[0.,1.],[1.,1.]])
    v=np.array([[1.,0.],[0.,2.],[3.,1.]])
    scores=q@q.T/np.sqrt(2)
    scores[np.triu_indices(3,1)]=-np.inf
    exp=np.exp(scores-np.max(scores,axis=-1,keepdims=True))
    weights=exp/exp.sum(-1,keepdims=True)
    output=weights@v
    np.testing.assert_allclose(output,[[1,0],[.3302384506733431,1.3395230986533138],[1.7587246087113846,1]],rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(weights[np.triu_indices(3,1)],[0,0,0])
    h=np.arange(12).reshape(1,3,4).reshape(1,3,2,2).transpose(0,2,1,3)
    np.testing.assert_array_equal(h[0,0],[[0,1],[4,5],[8,9]])
    np.testing.assert_array_equal(h[0,1],[[2,3],[6,7],[10,11]])
    np.testing.assert_array_equal(h.transpose(0,2,1,3).reshape(1,3,4),np.arange(12).reshape(1,3,4))
    assert np.arange(12).reshape(1,3,4)[0,1,0]==4
    assert np.arange(12).reshape(1,3,2,2)[0,1,0,0]==4
    assert h[0,0,1,0]==4
    row=np.array([1.,3.,5.,7.]);norm=(row-row.mean())/np.sqrt(row.var()+1e-5)
    np.testing.assert_allclose(norm,[-1.3416394449,-.4472131483,.4472131483,1.3416394449],atol=1e-9)
    np.testing.assert_allclose(np.array([[1.,2.],[3.,4.]])+[[.1,-.2],[.3,.4]],[[1.1,1.8],[3.3,4.4]])
    hidden=np.array([[1,-1]])@np.array([[1,0,1,1],[0,1,1,-1]])
    np.testing.assert_array_equal(hidden,[[1,-1,0,2]])
    np.testing.assert_array_equal(np.maximum(hidden,0)@np.array([[1,0],[0,1],[1,1],[1,-1]]),[[3,-2]])
    image=np.arange(1,17).reshape(4,4)
    patches=np.stack([image[r:r+2,c:c+2].reshape(-1) for r in (0,2) for c in (0,2)])
    np.testing.assert_array_equal(patches,[[1,2,5,6],[3,4,7,8],[9,10,13,14],[11,12,15,16]])
    projected=patches@np.array([[1,0],[0,1],[1,0],[0,1]])
    np.testing.assert_array_equal(projected,[[6,8],[10,12],[22,24],[26,28]])
    key_mask=np.zeros(34,dtype=bool);key_mask[10:18]=True
    logits=np.zeros(34);logits[key_mask]=-np.inf
    masked_weights=np.exp(logits)/np.exp(logits).sum()
    assert key_mask.sum()==8 and (~key_mask).sum()==26 and not key_mask[0]
    np.testing.assert_array_equal(masked_weights[key_mask],np.zeros(8))
    np.testing.assert_allclose(masked_weights[~key_mask],np.full(26,1/26))
    np.testing.assert_allclose(masked_weights.sum(),1)
    return {'status':'PASS','checks':['row-column contraction','bias broadcasting','reshape versus transpose','reduction axes','embedding gather','masked attention and value mixing','head split and exact inverse','LayerNorm row statistics','residual addition','expanded feedforward','paper storyboard products and cumulative sums','patch flattening and projection','34-slot missing-audio mask and renormalization'],
            'attention_output':output.tolist(),'normalization_output':norm.tolist(),
            'paper_storyboards':{'products':products.tolist(),'cumulative_sums':np.cumsum(products).tolist(),'patch_rows':patches.tolist(),'projected_patches':projected.tolist(),'source_slots':34,'masked_audio_slots':8,'visible_slots':26,'uniform_visible_weight':1/26},
            'scope':'Static paper diagrams and index mappings; no new model training or GPU performance claim.'}

if __name__=='__main__':
    result=verify()
    path=Path(__file__).with_name('tensor-walkthrough-verification.json')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
