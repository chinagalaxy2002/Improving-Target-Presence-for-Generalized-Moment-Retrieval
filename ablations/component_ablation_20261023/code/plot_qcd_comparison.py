"""Overlay A3 epoch04 on the existing E0/S4-Fusion Figure 7, without GPU/assets."""
import copy
import csv
import gzip
import json
import numpy as np
from environment import ROOT, read_rows, read_json, write_json, digest
from evaluation import Evaluation
from eval.metrics import compute_G_mIoU
from plot_utils import plt, axis, save

ORDER = ['E0', 'S4Fusion', 'S4Fusion_QCD_A3_ep04']
COLORS = ['#64748b', '#0072B2', '#d55e00', '#CC79A7']  # Added oracle color
MARKERS = ['o', 's', '^', 'D']
LINES = ['--', '-', '-.', ':']
NAMES = ['FlashVTG-GMR baseline', 'S4-Fusion / A0 ep04', 'S4-Fusion + QCD / A3 ep04']
ORACLE_GMIOU = 61.69  # Perfect-Existence Oracle (threshold-independent)


def read_gzip(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    import sys
    if '--render-only' in sys.argv:
        verification = read_json(ROOT/'reports/QCD_COMPARISON_VERIFICATION.json')
        for name, sources in verification['source_sha256'].items():
            for filename, expected in sources.items():
                assert digest(ROOT/'results'/name/filename) == expected
        def load_csv(filename):
            with (ROOT/'reports'/filename).open() as stream:
                return [dict(model=r['model'], **{k:float(v) for k,v in r.items() if k!='model'}) for r in csv.DictReader(stream)]
        rows = load_csv('threshold_grid_test_with_qcd.csv')
        summaries = load_csv('summary_test_tau05_with_qcd.csv')
        render({r['model']:r for r in summaries}, {n:[r for r in rows if r['model']==n] for n in ORDER})
        return
    gt = read_rows(ROOT/'data/test.jsonl')
    reports = {}
    grids = {}
    source_sha = {}
    all_rows = []
    subsets = []
    for name in ORDER:
        folder = ROOT/'results'/name
        pre = read_gzip(folder/'test_predictions_pre_nms.jsonl.gz')
        post = read_gzip(folder/'test_predictions.jsonl.gz')
        ev = Evaluation(pre, post, gt)
        cached = read_json(folder/'test_metrics.json')
        measured = ev.metrics(.5)
        assert all(measured[k] == cached['fixed_05'][k] for k in measured if k in cached['fixed_05'])
        ev.verify()
        summary = dict(measured, Rej_F1=200*measured['TN']/(2*measured['TN']+measured['FP']+measured['FN']))
        reports[name] = summary
        masks = {'worldcup2022':np.array([g['dataset_source']=='worldcup2022' for g in gt]),
                 'single':ev.count==1, 'multi':ev.count>1}
        for subset, mask in masks.items():
            subsets.append(dict(model=name, subset=subset, **ev.metrics(.5, mask=mask)))
        values = []
        for threshold in np.round(np.arange(0,1.001,.01),2):
            metric = ev.metrics(float(threshold))
            metric['Rej_F1'] = 200*metric['TN']/(2*metric['TN']+metric['FP']+metric['FN'])
            gated = copy.deepcopy(ev.post)
            for row,score in zip(gated,ev.score):
                if score < threshold:
                    row['pred_relevant_windows'] = []
            official = compute_G_mIoU(gated, gt, (1,3,5))
            assert all(metric[k] == official[k] for k in official), (name,threshold)
            values.append(metric)
            all_rows.append(dict(model=name, **metric))
        grids[name] = values
        source_sha[name] = {n:digest(folder/n) for n in ['test_predictions.jsonl.gz','test_predictions_pre_nms.jsonl.gz','test_metrics.json']}
        print(name, 'verified all 101 Test thresholds', flush=True)
    for filename, data in [('threshold_grid_test_with_qcd.csv',all_rows),('summary_test_tau05_with_qcd.csv',[dict(model=n,**reports[n]) for n in ORDER]),('subsets_test_tau05_with_qcd.csv',subsets)]:
        with (ROOT/'reports'/filename).open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(data[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(data)
    render(reports, grids)
    write_json(ROOT/'reports/QCD_COMPARISON_VERIFICATION.json',dict(passed=True,models=ORDER,
        Test_queries=len(gt),thresholds_per_model=101,official_G_checks=303,mAP_verified=True,
        checkpoint_epoch=4,Test_sweep_exploratory=True,source_sha256=source_sha))


def render(reports, grids):
    fig, axes = plt.subplots(2,3,figsize=(14.3,8.6))
    fig.subplots_adjust(left=.065,right=.985,bottom=.14,top=.865,wspace=.25,hspace=.34)
    axs = axes.ravel()
    for i,name in enumerate(ORDER):
        data = grids[name]
        t = [r['threshold'] for r in data]
        style = dict(color=COLORS[i],ls=LINES[i],marker=MARKERS[i],markevery=10,ms=4,lw=1.9)
        label = f'{NAMES[i]} (mAP: {reports[name]["mAP"]:.2f})'
        axs[0].plot(t,[r['G-mIoU@1'] for r in data],label=label,**style)
        axs[1].plot(t,[r['G-mIoU@3'] for r in data],**style)
        axs[2].plot(t,[r['G-mIoU@1']-b['G-mIoU@1'] for r,b in zip(data,grids['E0'])],**style)
        axs[3].plot([r['positive_recall'] for r in data],[r['N_acc'] for r in data],**style)
        axs[3].plot(reports[name]['positive_recall'],reports[name]['N_acc'],
                    marker=MARKERS[i],color=COLORS[i],ms=9,ls='None')
        axs[4].plot(t,[r['Rej_F1'] for r in data],**style)
        width=.25
        bars = axs[5].bar(np.arange(3)+(i-1)*width,[data[k]['G-mIoU@1'] for k in (40,50,60)],
                         width,color=COLORS[i],edgecolor='white')
        axs[5].bar_label(bars,fmt='%.2f',padding=3,fontsize=8,rotation=90)

    # Add Perfect-Existence Oracle line to panel (a)
    t_range = [r['threshold'] for r in grids['E0']]
    oracle_style = dict(color=COLORS[3],ls=LINES[3],marker=MARKERS[3],markevery=15,ms=3.5,lw=1.5,alpha=0.8)
    axs[0].plot(t_range, [ORACLE_GMIOU]*len(t_range),
                label=f'Perfect-Existence Oracle ({ORACLE_GMIOU:.2f})', **oracle_style)
    titles = ['(a) Threshold sensitivity','(b) Top-3 retrieval','(c) Relative to FlashVTG baseline',
              '(d) Rejection / positive retention','(e) Negative-class F1','(f) Fixed operating points']
    ylabs = ['G-mIoU@1 (%)','G-mIoU@3 (%)','Delta G-mIoU@1 (pp)','N-acc (%)','Rejection F1 (%)','G-mIoU@1 (%)']
    for i,ax in enumerate(axs):
        axis(ax,titles[i],'Positive recall (%)' if i==3 else 'Presence threshold',ylabs[i])
        if i in (0,1,2,4):
            ax.set_xlim(-.02,1.02);ax.set_xticks(np.arange(0,1.01,.2))
            ax.axvline(.5,color='#94a3b8',ls=':',lw=.8)
    axs[0].set_ylim(0,65);axs[1].set_ylim(0,65)
    axs[2].axhline(0,color='#64748b',lw=.8,ls=':')
    axs[3].set_xlim(-2,103);axs[3].set_ylim(-2,103)
    axs[4].set_ylim(0,100);axs[5].set_ylim(0,65)
    axs[5].set_xticks(np.arange(3),['0.4','0.5','0.6'])
    handles,labels = axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.995),ncol=1,frameon=False,fontsize=10)
    fig.text(.5,.042,'Test: 1,036 queries (544 positive / 492 negative). Large markers and dotted lines: threshold 0.50.\n'
             '101 thresholds per frozen model; mAP is positive-only post-NMS ranking without a hard presence gate.\n'
             'A3 epoch04 is the Val-selected QCD checkpoint; the completed epoch 4–10 Test sweep was exploratory.',
             ha='center',va='center',fontsize=9,color='#475569')
    save(fig,'fig7_threshold_sweep')


if __name__ == '__main__':
    main()
