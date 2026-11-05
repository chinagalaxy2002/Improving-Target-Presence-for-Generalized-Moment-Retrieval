import json
import numpy as np
import pytest
from soccergmr.backbones.registry import get_spec
from soccergmr.data import SoccerDataset,collate


def dataset(tmp_path,length=104):
    for name,dim in [('clip',512),('slowfast',2304)]:
        (tmp_path/name).mkdir()
        np.savez(tmp_path/name/'video.npz',features=np.ones((length,dim),np.float32))
    (tmp_path/'clip_text').mkdir()
    np.savez(tmp_path/'clip_text'/'qid7.npz',last_hidden_state=np.ones((32,512),np.float32),attention_mask=np.r_[np.ones(4),np.zeros(28)])
    label=tmp_path/'labels.jsonl'
    label.write_text(json.dumps(dict(qid=7,vid='video.mp4',duration=208.9,relevant_windows=[[100,110],[190,208.9]]))+'\n')
    config=dict(data=dict(splits={'test_full':str(label)},features={'full':str(tmp_path)},max_video_tokens=128,max_text_tokens=32))
    return SoccerDataset(config,'test_full',get_spec('sim_detr'))


def test_full_namespace_long_video_and_time_coordinates(tmp_path):
    d=dataset(tmp_path);item=d[0]
    assert len(item['video'])==104 and len(item['query'])==4
    assert item['spans'][0,0].item()==pytest.approx(105/208.9)
    assert item['masks'][1,-1]==1
    _,inputs,targets=collate([item])
    assert inputs['src_vid_mask'].sum()==104
    assert targets['mask_labels'][0].shape==(2,104)
    d.config['data']['max_video_tokens']=75
    with pytest.raises(ValueError,match='no silent truncation'):d[0]


def test_invalid_feature_is_not_silently_dropped(tmp_path):
    d=dataset(tmp_path)
    np.savez(tmp_path/'clip'/'video.npz',features=np.full((104,512),np.nan))
    with pytest.raises(ValueError,match='Invalid video features'):d[0]
