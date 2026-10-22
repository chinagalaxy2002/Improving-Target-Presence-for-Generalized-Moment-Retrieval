"""Exact cached curve computation and axes style from original backbone plots."""
from matplotlib.ticker import MultipleLocator
import numpy as np

def curves_for(scores,local,positive,thresholds):
 rows=[]
 for threshold in thresholds:
  accept=scores>float(threshold)
  tp=int((accept&positive).sum());tn=int((~accept&~positive).sum());fp=int((accept&~positive).sum());fn=int((~accept&positive).sum())
  contribution=np.where(accept[:,None],local,(~positive)[:,None])
  rows.append({'threshold':float(threshold),'G-mIoU@1':round(100*sum(contribution[:,0].tolist())/len(positive),2),'G-mIoU@3':round(100*sum(contribution[:,1].tolist())/len(positive),2),'N_acc':100*tn/int((~positive).sum()),'positive_recall':100*tp/int(positive.sum()),'Rej-F1':round(100*2*tn/(2*tn+fp+fn),2) if 2*tn+fp+fn else 0.,'TP':tp,'TN':tn,'FP':fp,'FN':fn})
 return rows

def axes_style(ax,xlabel,ylabel,tradeoff=False):
 ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.set_ylim(0,100 if tradeoff or ylabel.startswith('Rej') else 80)
 ax.set_xlim(0,100 if tradeoff else 1)
 ax.xaxis.set_major_locator(MultipleLocator(25 if tradeoff else .25))
 ax.yaxis.set_major_locator(MultipleLocator(25 if tradeoff or ylabel.startswith('Rej') else 20))
 ax.grid(True,alpha=.25,color='#94A3B8',linewidth=.6);ax.set_axisbelow(True)
