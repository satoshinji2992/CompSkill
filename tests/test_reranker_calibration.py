import torch
from skill_experience.retrieval.reranker_calibration import fit_bias, fit_residual, score_adjustment
from skill_experience.retrieval.reliability_key_compiler import fit_reliability_prior, fit_frozen_bias_keyopt


def inputs():
    torch.manual_seed(7)
    k=torch.nn.functional.normalize(torch.randn(4,8),dim=-1)
    q=torch.nn.functional.normalize(torch.randn(6,8),dim=-1)
    y=torch.tensor([[1.,0.,float('nan'),float('nan')],[0.,1.,1.,float('nan')],
                    [1.,1.,0.,float('nan')],[0.,0.,1.,float('nan')],
                    [1.,0.,0.,float('nan')],[0.,1.,0.,float('nan')]])
    return k,q,y,torch.isfinite(y)


def test_original_recipe_equivalence():
    k,q,y,m=inputs(); s=q@k.T
    old,_=fit_reliability_prior(k,q,y,m,steps=8)
    bias,_=fit_bias(s,q,y,m,steps=8)
    torch.testing.assert_close(bias,old.bias)
    index,_=fit_frozen_bias_keyopt(k,q,y,m,bias,steps=8,radius=.15)
    fit,_=fit_residual(s,q,y,m,bias,steps=8,radius=.15)
    torch.testing.assert_close(s+score_adjustment(q,fit),index.score(q))


def test_unknown_scores_and_zero_radius():
    k,q,y,m=inputs();s=q@k.T
    b,_=fit_bias(s,q,y,m,steps=8)
    t=s.clone();t[~m]=float('nan')
    other,_=fit_bias(t,q,y,m,steps=8)
    torch.testing.assert_close(b,other)
    fit,_=fit_residual(t,q,y,m,b,radius=0.)
    assert torch.count_nonzero(fit['delta'])==0
    assert b[3]==0
