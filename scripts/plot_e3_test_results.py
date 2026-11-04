"""Render current audited 11-backbone Test curves, not historical baselines."""
import csv,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parents[1];OUT=R/'results/e3_eleven'
NAMES={'moment_detr':'Moment-DETR [existing result reused]','eatr':'EaTR','qd_detr':'QD-DETR','cg_detr':'CG-DETR','sim_detr':'Sim-DETR','bm_detr':'BM-DETR','bam_detr':'BAM-DETR','uvcom':'UVCOM','tr_detr':'TR-DETR','taskweave':'TaskWeave','ld_detr':'LD-DETR'}
COLORS={'gmr':'#333333','geq_full':'#0072B2'};LABELS={'gmr':'GMR baseline','geq_full':'GEQ + aux'}
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 audit=json.loads((OUT/'AUDIT.json').read_text());assert audit['passed'] and len(audit['evaluations'])==44
 curves=list(csv.DictReader((OUT/'CURVES.csv').open()));metrics=list(csv.DictReader((OUT/'METRICS.csv').open()));assert len(curves)==4444
 data={}
 for n in NAMES:
  for split in ['test_standard','test_full']:
   for stage in ['gmr','geq_full']:
    rows=[x for x in curves if x['backbone']==n and x['stage']==stage and x['split']==split]
    assert [float(x['threshold']) for x in rows]==[i/100 for i in range(101)]
    m=next(x for x in metrics if x['backbone']==n and x['stage']==stage and x['split']==split)
    for k in [1,3]:assert abs(sum(float(rows[i]['G'+str(k)]) for i in [40,50,60])/3-float(m[f'G{k}_mean']))<1e-8
    data[n,split,stage]=(rows,m)
 def panel(ax,n,split,k,legend=False):
  for stage in ['gmr','geq_full']:
   rows,m=data[n,split,stage];ax.plot([float(x['threshold']) for x in rows],[float(x['G'+str(k)]) for x in rows],color=COLORS[stage],linestyle='-' if stage=='gmr' else '--',lw=1.8,label=f"{LABELS[stage]} | mAP {float(m['mAP']):.2f}")
   ax.scatter([.4,.5,.6],[float(rows[i]['G'+str(k)]) for i in [40,50,60]],s=12,color=COLORS[stage],zorder=3)
  ax.set_xlim(0,1);ax.set_ylim(0,100);ax.set_ylabel(f'G-mIoU@{k} (0–100)');ax.set_xlabel('Existence threshold');ax.grid(alpha=.23);ax.spines[['top','right']].set_visible(False)
  if legend:ax.legend(fontsize=9,loc='upper right',frameon=False)
  a=data[n,split,'gmr'][1];b=data[n,split,'geq_full'][1];dm=float(b['mAP'])-float(a['mAP']);dg=float(b[f'G{k}_mean'])-float(a[f'G{k}_mean'])
  ax.set_title(f"{NAMES[n]} | @ {k}",loc='left',fontsize=11,fontweight='bold')
  ax.text(.02,.94,f"ΔmAP {dm:+.2f} pp   Δmean G{k} {dg:+.2f} pp",transform=ax.transAxes,fontsize=8.5,va='top',color='#475569')
 combined=PdfPages(OUT/'BOTH_TESTS.pdf')
 for split,prefix,title in [('test_standard','STANDARD_TEST','Standard Test · 1,036 queries'),('test_full','FULL_TEST','Full-Decodable Test · 2,970 / 2,986 queries')]:
  fig,axes=plt.subplots(11,2,figsize=(15.5,31));fig.subplots_adjust(left=.065,right=.985,top=.95,bottom=.035,hspace=.63,wspace=.16)
  fig.suptitle('11-backbone fixed Test | '+title,fontsize=17,fontweight='bold',y=.986)
  fig.legend(handles=[Line2D([0],[0],color=COLORS[s],lw=2,linestyle='-' if s=='gmr' else '--',label=LABELS[s]) for s in ['gmr','geq_full']],loc='upper center',bbox_to_anchor=(.5,.976),ncol=2,frameon=False,fontsize=11)
  for row,n in enumerate(NAMES):
   for col,k in enumerate([1,3]):panel(axes[row,col],n,split,k,legend=True)
  fig.text(.5,.014,'Fixed Val-selected checkpoints; threshold grid 0:.01:1, score > threshold. mAP is ungated; Test not used for tuning.\nGMR baseline denotes the current auxiliary-trained Stage 1. Moment is reused with original source identities.',ha='center',fontsize=9,color='#475569')
  for ext in ['png','svg','pdf']:fig.savefig(OUT/(prefix+'.'+ext),dpi=160,facecolor='white')
  combined.savefig(fig);plt.close(fig)
  with PdfPages(OUT/(prefix+'_BY_BACKBONE.pdf')) as pages:
   for n in NAMES:
    fig,axes=plt.subplots(1,2,figsize=(13,4.5));fig.suptitle(NAMES[n]+' | '+title,fontsize=14,fontweight='bold')
    for ax,k in zip(axes,[1,3]):panel(ax,n,split,k,legend=True)
    fig.tight_layout(rect=[0,0,1,.93]);pages.savefig(fig);plt.close(fig)
 combined.close()
 files={p.name:digest(p) for p in OUT.iterdir() if p.suffix in ['.png','.svg','.pdf']}
 (OUT/'PLOT_AUDIT.json').write_text(json.dumps(dict(passed=True,source_metrics_sha256=digest(OUT/'METRICS.csv'),source_curves_sha256=digest(OUT/'CURVES.csv'),validated_series=44,threshold_points=101,curves_drawn=88,files=files),indent=2)+'\n')
 report=OUT/'REPORT.md';text=report.read_text();heading='## Fixed Test figures'
 if heading not in text:
  text+='\n\n'+heading+'\n\n![Standard Test](STANDARD_TEST.png)\n\n![Full Test](FULL_TEST.png)\n\n[Standard PDF](STANDARD_TEST.pdf) · [Full PDF](FULL_TEST.pdf) · [Combined PDF](BOTH_TESTS.pdf)\n\nCurrent GMR baseline is auxiliary-trained Stage1, not the historical no-aux baseline. CSV and original source provenance remain unchanged.\n'
  report.write_text(text)
 print('Rendered Standard / Full Test:11 backbones, G1/G3,101 thresholds, mAP legends. All44 CSV series validated.')
if __name__=='__main__':main()
