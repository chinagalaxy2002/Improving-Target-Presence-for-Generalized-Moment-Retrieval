from pathlib import Path
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
SPLIT_TEXT=""
NAMES=dict(moment_detr="Moment-DETR",eatr="EaTR",qd_detr="QD-DETR",cg_detr="CG-DETR")
def fmt(v,key): return f'{v:.4f}' if key.startswith('G-mIoU') else f'{v:.2f}'
def delta(v,b,key):
    d=round(v-b,4 if key.startswith('G-mIoU') else 2)+0.
    return d,('+' if d>0 else '')+fmt(d,key)

def table(stem,title,records,columns,pdf):
    widths=[110,260]+[78]*len(columns);total=sum(widths);h=116+34*len(records)+105
    fig=plt.figure(figsize=((total+40)/72,h/72),facecolor='white')
    ax=fig.add_axes([0,0,1,1]);ax.set_xlim(-20,total+20);ax.set_ylim(h,0);ax.axis('off')
    edges=[0]
    for w in widths:edges.append(edges[-1]+w)
    ax.text(0,24,title,fontsize=14,fontweight='bold')
    ax.text(0,45,SPLIT_TEXT,fontsize=9)
    ax.text(0,64,'Value above; (method - same-backbone GMR) below. G-mIoU: 0-1; others: percent / percentage points.',fontsize=9,color='#475569')
    headers=['Backbone','Method']+[k.replace('@tau=','\nτ=') for k in columns]
    for j,k in enumerate(headers):ax.text(edges[j]+3 if j<2 else (edges[j]+edges[j+1])/2,92,k,ha='left' if j<2 else 'center',va='center',fontsize=8.2,fontweight='bold')
    ax.plot([0,total],[116,116],color='#172b3a')
    bases={r['backbone']:r for r in records if r['method']=='gmr'}
    tex=[r'\noindent\textbf{'+title.replace('_',r'\_')+r'}\par',r'\resizebox{\textwidth}{!}{\begin{tabular}{ll'+'r'*len(columns)+r'}\toprule',
         'Backbone & Method & '+' & '.join(k.replace('+',r'$^{+}$').replace('@tau=',r' $\tau=$').replace('%',r'\%') for k in columns)+r' \\\midrule']
    last=None
    for i,r in enumerate(records):
        y=116+34*i;base=bases[r['backbone']];isbase=r['method']=='gmr'
        if isbase:ax.add_patch(plt.Rectangle((0,y),total,34,facecolor='#e2edf8',edgecolor='none'))
        elif list(NAMES).index(r['backbone'])%2:ax.add_patch(plt.Rectangle((0,y),total,34,facecolor='#f1f5f9',edgecolor='none'))
        if last and last!=r['backbone']:ax.plot([0,total],[y,y],color='#64748b',lw=.6);tex.append(r'\midrule')
        label=r['label']+(' (baseline)' if isbase else '')
        vals=[NAMES[r['backbone']] if last!=r['backbone'] else '',label]
        for j,s in enumerate(vals):ax.text(edges[j]+3,y+17,s,fontsize=8.7,va='center',fontweight='bold' if isbase else 'normal')
        texvals=vals[:]
        for j,k in enumerate(columns,2):
            v=r['values'][k];b=base['values'][k];value=fmt(v,k)
            ax.text(edges[j+1]-6,y+17 if isbase else y+10,value,ha='right',va='center',fontsize=8.5,fontweight='bold' if isbase else 'normal')
            if isbase:texvals.append(r'\textbf{'+value+'}')
            else:
                d,ds=delta(v,b,k);color='#15803d' if d>0 else '#b91c1c' if d<0 else '#64748b'
                ax.text(edges[j+1]-6,y+25,'('+ds+')',ha='right',va='center',fontsize=7.5,color=color)
                tc='Gain' if d>0 else 'Loss' if d<0 else 'Zero'
                texvals.append(r'\shortstack{'+value+r' \\ {\scriptsize\textcolor{'+tc+'}{('+ds+')}}}')
        tex.append(' & '.join(texvals)+r' \\');last=r['backbone']
    end=116+34*len(records);ax.plot([0,total],[end,end],color='#172b3a')
    ax.text(0,end+24,'Green: gain',color='#15803d',fontsize=9)
    ax.text(105,end+24,'Red: decrease',color='#b91c1c',fontsize=9)
    ax.text(225,end+24,'Gray: unchanged; all deltas are relative to GMR.',color='#64748b',fontsize=9)
    ax.text(0,end+45,'N-acc = TN / negatives. τ is the existence threshold. Localization uses positive ungated candidates.',fontsize=9)
    ax.text(0,end+65,'Full dataset and GEQ Full training mode are distinct. Coverage and temporal limitations: see DATA.md.',fontsize=9)
    ax.text(0,end+85,'Same Standard-trained checkpoints across both Test domains. Training budgets and native seeds differ; see TRAINING.md.',fontsize=8.5)
    fig.savefig(HERE/(stem+'.png'),dpi=150)
    fig.savefig(HERE/(stem+'.pdf'),metadata={'CreationDate':None,'ModDate':None});pdf.savefig(fig);plt.close(fig)
    tex += [r'\bottomrule\end{tabular}}\par',r'G-mIoU: 0--1. Others: percent. Parentheses: method minus same-backbone GMR.\par']
    return '\n'.join(tex)
