"""CPU regression check using a real positive annotation with zero assigned points."""
import math,torch
from paths import *
from native_factory import TargetAdapter

def main():
 torch.set_num_threads(2);ad=TargetAdapter('flashvtg','cpu','charades',seed=2023)
 data=rows(OUT/'data/train.jsonl');row=next(r for r in data if r['evaluation_id']=='train:005243')
 n=json.loads((OUT/'data/audit.json').read_text())['video_lengths'][row['vid']]
 point=ad.base.generator([torch.zeros(1,math.ceil(n/s),256) for s in (1,2,4,8)])
 cls,reg=ad.criterion.loss.get_target_single(point,torch.tensor(row['relevant_windows'])*.5,torch.ones(1,1,dtype=torch.long));assert not cls.any()
 src=torch.ones_like(reg[None],requires_grad=True);output={};ad.criterion.loss.loss_reg({'out_coord':src},output,cls[None],reg[None]);value=output['loss_reg'];assert value.item()==0 and value.requires_grad
 value.backward();assert torch.isfinite(src.grad).all() and not src.grad.any()
 # Ordinary positive regression must still use the unchanged native loss.
 cls,reg=ad.criterion.loss.get_target_single(point,torch.tensor([[0.,3.]]),torch.ones(1,1,dtype=torch.long));assert cls.any()
 src=torch.ones_like(reg[None],requires_grad=True);output={};ad.criterion.loss.loss_reg({'out_coord':src},output,cls[None],reg[None]);mask=cls[None,:,None].expand(-1,-1,2).bool();expected=ad.criterion.loss._loss_reg(src,reg[None],weight=mask,avg_factor=mask.sum());assert torch.equal(output['loss_reg'],expected)
 dump(OUT/'reports/flash_zero_point_guard.json',dict(passed=True,evaluation_id=row['evaluation_id'],vid=row['vid'],real_positive_zero_point_loss=0.,gradient_finite_and_zero=True,ordinary_positive_regression_unchanged=True))
 print('REAL_ZERO_POINT_REGRESSION_GUARD_PASS')
if __name__=='__main__':main()
