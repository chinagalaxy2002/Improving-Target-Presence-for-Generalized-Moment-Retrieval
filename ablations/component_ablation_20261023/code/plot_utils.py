"""Common figure style: dual encoding, vector output, 300-DPI previews."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
# CJK is optional for regeneration, but the full user-specified method name is retained.
for p in [Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]:
 if p.exists():font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':['DejaVu Sans','Noto Sans CJK JP'],'font.size':10,'axes.titlesize':11,'axes.titleweight':'bold','axes.titlecolor':'#0f172a','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
ORDER=['E0','S4Fusion']
COLORS={'E0':'#64748b','S4Fusion':'#0072B2'}
LINES={'E0':'--','S4Fusion':'-'}
MARKERS={'E0':'o','S4Fusion':'s'}
LABELS={'E0':'FlashVTG-GMR baseline','S4Fusion':'S4-Fusion (no candidate fusion)'}
def axis(ax,title,xlabel,ylabel,ylim=None):
 ax.set_title(title,loc='left',pad=10);ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.grid(axis='y',color='#cbd5e1',alpha=.6,lw=.6);ax.set_axisbelow(True)
 if ylim is not None:ax.set_ylim(*ylim)
def legend(fig,ax):
 h,l=ax.get_legend_handles_labels();fig.legend(h,l,loc='outside upper center',ncol=2,frameon=False,fontsize=9)
def save(fig,name):
 (ROOT/'figures').mkdir(exist_ok=True)
 for ext in ['png','pdf','svg']:fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=300,bbox_inches='tight')
 svg=ROOT/'figures'/f'{name}.svg'
 svg.write_text('\n'.join(l.rstrip() for l in svg.read_text().splitlines())+'\n')
 plt.close(fig)
