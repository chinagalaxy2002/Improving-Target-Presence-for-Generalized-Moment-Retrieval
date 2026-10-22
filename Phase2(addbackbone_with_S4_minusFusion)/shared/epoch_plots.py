"""Test-only fixed-epoch S4 curves, preserving original/GMR comparison lines."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from official_eval.eval_main import evaluate_gmr
from official_eval.metrics import get_existence_score, _compute_set_iou_score, prepare_submission_for_gmiou, compute_G_mIoU, compute_gmr_cls
from plot_reference import curves_for, axes_style

ROOT = Path(__file__).resolve().parents[1]
FIGURES = Path('/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/figures')
LABELS = Path('/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/data/label/Standard/test.jsonl')
NAMES = {'moment_detr':'Moment-DETR','qd_detr':'QD-DETR','cg_detr':'CG-DETR','eatr':'EaTR'}
COLORS = {'original':'#0072B2','gmr':'#D55E00','epoch_003':'#009E73','epoch_004':'#CC79A7','epoch_005':'#56B4E9','epoch_006':'#333333'}
MARKERS = {'original':'o','gmr':'s','epoch_003':'^','epoch_004':'v','epoch_005':'D','epoch_006':'P'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label(key):
    return {'original':'Original','gmr':'GMR'}.get(key,'S4 epoch '+str(int(key.rsplit('_',1)[-1])) if key.startswith('epoch_') else key)


def measure(predictions, groundtruth, backbone, epoch, checkpoint, predictions_path):
    expected = [json.loads(line) for line in LABELS.read_text().splitlines() if line.strip()]
    assert len(expected) == len(groundtruth) == len(predictions) == 1036
    expected_by_id = {str(g['qid']):g for g in expected}
    actual_by_id = {str(g['qid']):g for g in groundtruth}
    by_id = {str(p['qid']):p for p in predictions}
    assert len(by_id) == 1036 and set(by_id) == set(expected_by_id) == set(actual_by_id)
    assert all(actual_by_id[k]['relevant_windows'] == g['relevant_windows'] for k,g in expected_by_id.items())
    aligned = [{**by_id[str(g['qid'])], 'qid':g['qid']} for g in expected]
    scores = np.array([get_existence_score(p)[0] for p in aligned])
    positive = np.array([bool(g['relevant_windows']) for g in expected])
    clean,_ = prepare_submission_for_gmiou(aligned,cls_threshold=-1,max_pred_windows=10)
    local = np.array([[_compute_set_iou_score([w[:2] for w in p['pred_relevant_windows'][:k]],g['relevant_windows']) for k in (1,3)] for p,g in zip(clean,expected)])
    grid = curves_for(scores,local,positive,np.round(np.linspace(0,1,101),2))
    empirical = curves_for(scores,local,positive,np.unique(np.r_[0.,scores,1.]))
    for threshold in (0.,.4,.6,.8,1.):
        row = next(r for r in grid if r['threshold']==threshold)
        gated,_ = prepare_submission_for_gmiou(aligned,cls_threshold=threshold,max_pred_windows=10)
        official_g = compute_G_mIoU(gated,expected,k_list=(1,3))
        official_c = compute_gmr_cls(aligned,expected,thresholds=(threshold,))['per_threshold'][str(threshold)]
        assert all(row[k] == official_g[k] for k in ('G-mIoU@1','G-mIoU@3'))
        assert row['Rej-F1'] == official_c['Rej-F1']
    metrics = evaluate_gmr(aligned,expected,cls_thresholds=(.4,.5,.6,.8),map_num_workers=0,verbose=False)
    return {'backbone':backbone,'epoch':epoch,'split':'test','queries':1036,'checkpoint':str(checkpoint),
            'checkpoint_sha256':sha(checkpoint),'predictions':str(predictions_path),'predictions_sha256':sha(predictions_path),
            'test_labels_sha256':sha(LABELS),'grid':grid,'empirical_tradeoff':empirical,
            'official_metrics':metrics,'checkpoint_selection':'none; every epoch is retained',
            'threshold_selection':'none; fixed-grid sensitivity curves only'}


def panel(ax,kind,series):
    if kind == 'gmiou':
        axes_style(ax,r'Existence threshold $\tau$','G-mIoU (%)')
        for key,data in series.items():
            rows = data['grid']
            for k,style in ((1,'-'),(3,'--')):
                ax.plot([r['threshold'] for r in rows],[r[f'G-mIoU@{k}'] for r in rows],color=COLORS[key],linestyle=style,
                        marker=MARKERS[key],markevery=20,markersize=3,linewidth=1.65,label=f'{label(key)}, k={k}')
                ax.scatter([.4],[rows[40][f'G-mIoU@{k}']],color=COLORS[key],marker=MARKERS[key],s=23,zorder=5)
        ax.axvline(.4,color='#64748B',linestyle=':',linewidth=.8)
    elif kind == 'tradeoff':
        axes_style(ax,'Positive recall (%)','N-acc (%)',tradeoff=True)
        for key,data in series.items():
            rows=data['empirical_tradeoff']; point=data['grid'][40]
            ax.plot([r['positive_recall'] for r in rows],[r['N_acc'] for r in rows],color=COLORS[key],linestyle='--' if key=='original' else '-',linewidth=1.65,label=label(key))
            ax.scatter([point['positive_recall']],[point['N_acc']],color=COLORS[key],marker=MARKERS[key],s=25,zorder=5)
    else:
        axes_style(ax,r'Existence threshold $\tau$','Rej-F1 (%)')
        for key,data in series.items():
            rows=data['grid']
            ax.plot([r['threshold'] for r in rows],[r['Rej-F1'] for r in rows],color=COLORS[key],linestyle='--' if key=='original' else '-',
                    marker=MARKERS[key],markevery=20,markersize=3,linewidth=1.65,label=label(key))
            ax.scatter([.4],[rows[40]['Rej-F1']],color=COLORS[key],marker=MARKERS[key],s=23,zorder=5)
        ax.axvline(.4,color='#64748B',linestyle=':',linewidth=.8)
    handles,labels=ax.get_legend_handles_labels()
    ax.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.24),frameon=False,
              fontsize=7.5,ncol=2 if len(labels)<=6 else 3,columnspacing=.8,handlelength=1.7)


def render(backbone,series,directory):
    directory.mkdir(parents=True,exist_ok=False)
    def save(fig,stem):
        for fmt in ('png','pdf','svg'):fig.savefig(directory/f'{backbone}_{stem}.{fmt}',dpi=220,bbox_inches='tight',pad_inches=.1)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,5.6))
    fig.subplots_adjust(left=.055,right=.985,bottom=.32,top=.84,wspace=.34)
    for ax,kind,title in zip(axes,('gmiou','tradeoff','rej_f1'),('(a) Generalized retrieval','(b) Rejection / retention','(c) Null-query rejection')):
        panel(ax,kind,series);ax.set_title(title)
    fig.suptitle(NAMES[backbone]+' | Soccer-GMR Standard test (1036 queries)',fontsize=13)
    fig.text(.5,.01,'Original and GMR baseline curves retained; S4 uses fixed epoch checkpoints. Markers: threshold 0.4.',ha='center',fontsize=9)
    save(fig,'analysis')
    for kind in ('gmiou','tradeoff','rej_f1'):
        fig,ax=plt.subplots(figsize=(6.2,5.2));fig.subplots_adjust(bottom=.31)
        panel(ax,kind,series);ax.set_title(NAMES[backbone]+' | Standard test');save(fig,kind)
    return len(series)*2


def generate(backbone,epoch,predictions,groundtruth,checkpoint,predictions_path):
    folder=FIGURES/backbone/'finetune_epochs';folder.mkdir(exist_ok=True)
    epochdir=folder/f'epoch_{epoch:03d}';epochdir.mkdir(exist_ok=False)
    data=measure(predictions,groundtruth,backbone,epoch,checkpoint,predictions_path)
    (epochdir/'S4_TEST_CURVES.json').write_text(json.dumps(data,indent=2)+'\n')
    baseline={v:json.loads((ROOT/'shared/plot_baselines'/f'{backbone}_{v}.json').read_text()) for v in ('original','gmr')}
    assert all(d['checks']['query_coverage']==1036 for d in baseline.values())
    one={**baseline,f'epoch_{epoch:03d}':data}
    cumulative=dict(baseline)
    for previous in range(3,epoch+1):
        path=folder/f'epoch_{previous:03d}/S4_TEST_CURVES.json'
        if path.exists():cumulative[f'epoch_{previous:03d}']=json.loads(path.read_text())
    render(backbone,one,epochdir/'this_epoch')
    render(backbone,cumulative,epochdir/'all_epochs_so_far')
    manifest={'backbone':backbone,'epoch':epoch,'split':'test','checkpoint':str(checkpoint),'checkpoint_sha256':data['checkpoint_sha256'],
              'per_epoch_gmiou_lines':6,'cumulative_gmiou_lines':len(cumulative)*2,'original_baseline_lines_retained':4,
              'figure_types':['analysis','gmiou','tradeoff','rej_f1'],'formats':['png','pdf','svg'],
              'this_epoch':str(epochdir/'this_epoch'),'all_epochs_so_far':str(epochdir/'all_epochs_so_far')}
    (epochdir/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    latest=folder/'latest';temporary=folder/'.latest.tmp'
    temporary.unlink(missing_ok=True);temporary.symlink_to(epochdir.name+'/all_epochs_so_far',target_is_directory=True);temporary.replace(latest)
    links=['# Fixed-epoch S4 test comparisons','',
           'Each epoch keeps all four Original/GMR G-mIoU lines and adds S4 k=1/k=3. Curves use Standard test; no best-epoch selection.', '']
    for n in range(3,7):
        if (folder/f'epoch_{n:03d}/FIGURE_MANIFEST.json').exists():
            links.append(f'- Epoch {n}: [this epoch](epoch_{n:03d}/this_epoch/{backbone}_analysis.png), [all epochs so far](epoch_{n:03d}/all_epochs_so_far/{backbone}_analysis.png).')
    (folder/'README.md').write_text('\n'.join(links)+'\n')
    return manifest,data['official_metrics']
