import random
import numpy as np
import pytest
import torch
from soccergmr import checkpoint as ck


def test_resume_reproduces_optimizer_and_next_epoch_order(tmp_path):
    ck.seed_all(2023);model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    generator=torch.Generator().manual_seed(2023)
    x=torch.randn(5,2);y=torch.randn(5,1)
    def step():
        order=torch.randperm(5,generator=generator)
        loss=((model(x[order])-y[order])**2).mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        return order
    step()
    ck.save(tmp_path/'last.pt',dict(identity='a',stage='gmr',model=model.state_dict(),optimizer=optimizer.state_dict(),rng=ck.rng_state(generator)))
    expected_order=step();expected={k:v.clone() for k,v in model.state_dict().items()}
    saved=ck.load(tmp_path/'last.pt','a','gmr')
    model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);ck.restore_rng(saved['rng'],generator)
    assert torch.equal(step(),expected_order)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in expected.items())
    with pytest.raises(ValueError,match='identity mismatch'):ck.load(tmp_path/'last.pt','different')
    with pytest.raises(ValueError,match='Wrong checkpoint stage'):ck.load(tmp_path/'last.pt',stage='geq_full')
