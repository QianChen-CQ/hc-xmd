#!/usr/bin/env python3
"""HC-XMD minor figure revision: original matplotlib palette and chart forms.

Run: python build_figures.py [--data figures_data.csv] [--out OUTPUT_DIRECTORY]
Requires Python 3, numpy and matplotlib. Figure 1 is intentionally untouched.
"""
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=HERE/'figures_data.csv');p.add_argument('--out',type=Path,default=HERE)
p.add_argument('--figure',choices=['all','2'],default='all',help='Regenerate only Fig.2 or all statistical figures.')
args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
with args.data.open(encoding='utf-8-sig') as f: DATA=list(csv.DictReader(f))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':9.2,
    'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
    'axes.linewidth':.8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'savefig.dpi':220,'axes.spines.top':True,'axes.spines.right':True})
BLUE='#1f77b4';ORANGE='#ff7f0e';GREEN='#2ca02c';RED='#d62728';PURPLE='#9467bd';GRAY='#888888'
METHODS=[('RegretNet_PaperReimplementation','RegretNet [RI]',BLUE,'o'),
         ('AMenuNet_PaperReimplementation','AMenuNet [RI]',ORANGE,'s'),
         ('RegretNet_OptimalerBudget','RegretNet [A]',BLUE,'^'),
         ('RegretFormer_AuthorsCheckpoint','RegretFormer [A]',PURPLE,'D')]
SHORT={'RegretNet_PaperReimplementation':'RegretNet [RI]',
       'AMenuNet_PaperReimplementation':'AMenuNet [RI]',
       'RegretNet_OptimalerBudget':'RegretNet [A]',
       'RegretFormer_AuthorsCheckpoint':'RegretFormer [A]'}

def rows(phase,method=None,setting=None,audit='strong'):
    return sorted([r for r in DATA if r['input_phase']==str(phase)
       and (method is None or r['method']==method)
       and (setting is None or r['setting']==setting)
       and (audit is None or r['audit']==audit)],key=lambda r:int(r['seed']))
def vv(r,k):return np.array([float(i[k]) for i in r])
def grid(ax,axis='both'):
    ax.grid(axis=axis,alpha=.20);ax.set_axisbelow(True);ax.tick_params(length=3,width=.7)
def save(fig,name):
    for ext in ['pdf','png','svg']:fig.savefig(args.out/f'{name}.{ext}',bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
def seed_legend(fig,y=.02):
    hs=[Line2D([],[],marker=mk,color=GRAY,linestyle='none',label=f'Seed {i}',markersize=4) for i,mk in enumerate(['o','s','^'])]
    hs.append(Line2D([],[],marker='D',color=BLUE,linestyle='none',label='Three-seed mean',markersize=4))
    fig.legend(handles=hs,loc='lower center',bbox_to_anchor=(.5,y),frameon=False,ncol=4,handletextpad=.4)

def phase1():
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.5))
    methods=['RegretNet_PaperReimplementation','AMenuNet_PaperReimplementation']
    params=[float(rows(1,m)[0]['parameter_count']) for m in methods]
    ax=axs[0];ax.bar([0,1],params,color=[BLUE,ORANGE],width=.5)
    for i,y in enumerate(params):ax.text(i,y+23000,f'{y:,.0f}',ha='center',fontsize=8)
    ax.set_ylim(0,920000);ax.set_yticks([0,400000,800000],['0','400,000','800,000'])
    ax.set_xticks([0,1],['RegretNet','AMenuNet']);ax.set_ylabel('Count');ax.set_title('(a) Parameter count');grid(ax,'y')
    ax=axs[1];ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_axis_off()
    ax.set_title('(b) Mechanism representation')
    slots=int(float(rows(1,methods[1])[0]['menu_size']))
    for y,color in [(.53,BLUE),(.05,ORANGE)]:
        ax.add_patch(Rectangle((.01,y),.98,.41,fill=False,edgecolor=color,lw=1.1))
    ax.text(.5,.84,'RegretNet: direct neural map',ha='center',color=BLUE,fontweight='bold',fontsize=8)
    ax.text(.5,.67,'Bids → allocation and payment\nNo fixed candidate-menu list',ha='center',va='center',fontsize=7.7)
    ax.text(.5,.36,'AMenuNet: explicit menu',ha='center',color=ORANGE,fontweight='bold',fontsize=8)
    ax.text(.5,.19,f'{slots} learned options + no-sale\nMenu selection + pivot payment',ha='center',va='center',fontsize=7.7)
    fig.subplots_adjust(left=.115,right=.975,top=.88,bottom=.23,wspace=.42)
    fig.text(.5,.025,'Phase I: static 3 × 10. Both methods were trained and evaluated; representations do not measure human burden.',ha='center',fontsize=7.6)
    save(fig,'fig2_phase1_structure')

def static_scatter():
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.1))
    for i,(ax,setting) in enumerate(zip(axs,['3x10','2x2'])):
        for m,label,c,mk in METHODS:
            rr=rows(2,m,setting)
            if not rr:continue
            x=vv(rr,'regret_mean');y=vv(rr,'revenue')
            ax.scatter(x,y,marker=mk,s=22,facecolors='white',edgecolors=c,linewidths=.8,zorder=3)
            ax.errorbar(x.mean(),y.mean(),xerr=x.std(ddof=1),yerr=y.std(ddof=1),fmt=mk,color=c,ms=5,capsize=2,lw=.8,zorder=4)
        ax.set_title(f'({chr(97+i)}) Static '+setting.replace('x',' × '))
        ax.set_xlabel('Strong empirical regret mean');ax.set_ylabel('Revenue mean');grid(ax)
    axs[0].set_xlim(-.0007,.014);axs[0].set_ylim(5.45,6.35)
    axs[1].set_xlim(-.00015,.0027);axs[1].set_ylim(.86,.935)
    axs[0].set_xticks([0,.005,.01]);axs[1].set_xticks([0,.001,.002])
    hh=[Line2D([],[],marker=mk,color=c,linestyle='none',markersize=5,label=label) for _,label,c,mk in METHODS]
    fig.legend(handles=hh,loc='lower center',bbox_to_anchor=(.5,.055),ncol=4,frameon=False,columnspacing=1,handletextpad=.4)
    fig.subplots_adjust(left=.09,right=.985,top=.89,bottom=.29,wspace=.32)
    fig.text(.5,.027,'[RI] paper reimplementation; [A] released author checkpoint. Open marks: individual units; bars: sample SD.',ha='center',fontsize=7.7)
    fig.text(.5,-.012,'Settings and source roles remain separate. GemNet all-null selected checkpoints are diagnostic only.',ha='center',fontsize=7.7)
    save(fig,'fig3_phase2_static_tradeoffs')

def static_diagnostics():
    fig,axs=plt.subplots(2,2,figsize=(7.1,4.1))
    for r,setting in enumerate(['3x10','2x2']):
        methods=[m[0] for m in METHODS if rows(2,m[0],setting)]
        labels=[SHORT[m] for m in methods];ys=np.arange(len(methods))
        for c,metric in enumerate(['regret_max','pmr']):
            ax=axs[r,c]
            for y,m in enumerate(methods):
                vals=vv(rows(2,m,setting),metric);mean=vals.mean()
                ax.hlines(y,0,mean,color=BLUE,lw=1.2)
                ax.scatter([mean],[y],s=23,color=BLUE,zorder=3)
                ax.scatter(vals,np.array([y-.09,y,y+.09]),s=14,facecolors='white',edgecolors=BLUE,lw=.65,zorder=4)
            ax.set_yticks(ys,labels);ax.set_ylim(-.45,len(methods)-.55);ax.invert_yaxis()
            ax.set_title(('(a) ' if (r,c)==(0,0) else '(b) ' if (r,c)==(0,1) else '(c) ' if c==0 else '(d) ')+setting.replace('x',' × ')+' — '+('Regret maximum' if c==0 else 'PMR'))
            if c==0:
                xmax=max(vv(rows(2,m,setting),metric).max() for m in methods)
                ax.set_xlim(-xmax*.04,xmax*1.09);ax.set_xlabel('Maximum detected gain')
            else:
                ax.set_xlim(-.04,1.08);ax.set_xticks([0,.5,1],['0%','50%','100%']);ax.set_xlabel('Bidder–profile fraction')
            grid(ax,'x')
    fig.subplots_adjust(left=.19,right=.985,top=.91,bottom=.18,wspace=.77,hspace=.8)
    fig.text(.5,.065,'Filled point: mean across 3 units; open points: individual units. Means of maxima are not pooled maxima.',ha='center',fontsize=7.7)
    fig.text(.5,.025,'Strong audit; PMR counts bidder–profile gains > 10⁻⁶, not observed human misreporting.',ha='center',fontsize=7.7)
    save(fig,'fig4_audit_sensitivity')

TASKS=[('Context · 3 × 10','CITransNetNoContext','CITransNet',['No context','CITransNet']),
 ('Preference · 2 × 2','RegretNetPreferenceBaseline','PreferenceNet',['Matched control','PreferenceNet']),
 ('Fairness · 2 × 2','RegretNetProportionBaseline','ProportionNet',['Matched control','ProportionNet'])]
MARKERS=['o','s','^']

def phase3_performance():
    fig,axs=plt.subplots(2,3,figsize=(7.1,4.0))
    for c,(title,baseline,method,labels) in enumerate(TASKS):
        ax=axs[0,c]
        for m,color in [(baseline,GRAY),(method,BLUE)]:
            rr=rows(3,m);x=vv(rr,'regret_mean');y=vv(rr,'revenue')
            for s in range(3):ax.scatter(x[s],y[s],s=26,marker=MARKERS[s],color=color,zorder=3)
            ax.scatter(x.mean(),y.mean(),s=32,marker='D',facecolors='white',edgecolors=color,lw=1,zorder=4)
        ax.set_title(f'({chr(97+c)}) {title}',fontsize=8.7);ax.set_ylabel('Revenue');grid(ax)
        scale=1e-4 if c==0 else 1e-3
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos,s=scale:f'{value/s:g}'))
        ax.set_xlabel('Mean regret ($10^{-4}$ units)' if c==0 else 'Mean regret ($10^{-3}$ units)',fontsize=8)
        if c==0:
            ax.set_ylim(-.1,3.3);ax.set_xlim(-.00003,.00045)
            ax.annotate('Control seed 0',xy=(0,0),xytext=(.00008,.55),fontsize=6.9,arrowprops={'arrowstyle':'-','lw':.6})
        else:ax.set_ylim(.82,.965)
        ax=axs[1,c]
        for j,m in enumerate([baseline,method]):
            v=vv(rows(3,m),'regret_max')
            ax.hlines(j,0,v.mean(),color=BLUE,lw=1.2)
            ax.scatter([v.mean()],[j],s=24,color=BLUE,zorder=3)
            for s in range(3):ax.scatter(v[s],j+[-.085,0,.085][s],s=20,marker=MARKERS[s],facecolors='white',edgecolors=BLUE,lw=.7,zorder=4)
        ax.set_yticks([0,1],['Control','Method']);ax.invert_yaxis();ax.set_ylim(1.4,-.4)
        ax.set_xlabel('Strong regret maximum');ax.set_title('Incentive diagnostic',fontsize=8.5);grid(ax,'x')
        xmax=max(vv(rows(3,m),'regret_max').max() for m in [baseline,method]);ax.set_xlim(-.035*xmax,xmax*1.08)
    fig.subplots_adjust(left=.09,right=.985,top=.91,bottom=.24,wspace=.72,hspace=.75)
    hs=[Line2D([],[],marker='o',color=GRAY,linestyle='none',label='Matched control',markersize=4),Line2D([],[],marker='o',color=BLUE,linestyle='none',label='Task method',markersize=4),Line2D([],[],marker='D',color=BLUE,mfc='white',linestyle='none',label='Three-seed mean (top)',markersize=4)]
    fig.legend(handles=hs,loc='lower center',bbox_to_anchor=(.5,.087),ncol=3,frameon=False,handletextpad=.4)
    fig.text(.5,.047,'Circles / squares / triangles identify seeds 0 / 1 / 2. The collapsed no-context seed remains included.',ha='center',fontsize=7.7)
    fig.text(.5,.009,'Each column compares its own task and control; axes differ. There is no cross-task ranking.',ha='center',fontsize=7.7)
    save(fig,'fig5_phase3_matched_performance')

def phase3_heatmap():
    fig,axs=plt.subplots(1,3,figsize=(7.1,2.75))
    metrics=['welfare','pca','paper_unfairness_sum'];titles=['Context: welfare ↑','Synthetic preference pass ↑','Fairness: unfairness ↓']
    limits=[4.,1.,.3]
    for c,(title,b,m,labels) in enumerate(TASKS):
        ax=axs[c];arr=np.array([vv(rows(3,i),metrics[c]) for i in [b,m]])
        im=ax.imshow(arr,cmap='Blues',vmin=0,vmax=limits[c],aspect='auto')
        ax.set_xticks([0,1,2],['Seed 0','Seed 1','Seed 2']);ax.set_yticks([0,1],['Control','Method'])
        ax.set_title(f'({chr(97+c)}) '+titles[c],fontsize=8.45)
        for r in range(2):
            for k in range(3):
                v=arr[r,k]
                txt=(f'{v*100:.1f}%' if c==1 else f'{v:.2e}' if v<1e-6 else f'{v:.4f}' if c==2 else f'{v:.3f}')
                ax.text(k,r,txt,ha='center',va='center',color='white' if v/limits[c]>.56 else 'black',fontsize=7.8)
        cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.085,pad=.27,aspect=17)
        if c==1:cb.set_ticks([0,.5,1],labels=['0%','50%','100%'])
        if c==0:cb.set_ticks([0,2,4])
        if c==2:cb.set_ticks([0,.15,.3])
        cb.ax.tick_params(labelsize=7.6,length=2)
    fig.subplots_adjust(left=.073,right=.985,top=.82,bottom=.23,wspace=.43)
    fig.text(.5,.072,'Raw values; each panel has its own units and color scale. The same blue shade does not imply equal quality.',ha='center',fontsize=7.7)
    fig.text(.5,.026,'Pass rate is rule-defined; positive-part unfairness differs from absolute TVF. All 3 seeds are retained.',ha='center',fontsize=7.7)
    save(fig,'fig6_phase3_task_metrics')

phase1()
if args.figure=='all':
    static_scatter();static_diagnostics();phase3_performance();phase3_heatmap()
(args.out/'render_manifest.json').write_text(json.dumps({'data_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),'matplotlib_version':matplotlib.__version__,'figure_numbers':[2] if args.figure=='2' else [2,3,4,5,6],'figure1':'original retained by manuscript editor','data_rows':len(DATA),'palette':{'RegretNet':BLUE,'AMenuNet':ORANGE,'GemNet':GREEN,'audit_variant_original':RED,'RegretFormer':PURPLE,'heatmap':'Blues'},'constraint':'No fabricated human scores, no cross-task normalization, no erased collapsed seed.'},indent=2))
print('Created Figure 2 only.' if args.figure=='2' else 'Created figures 2–6; Figure 1 left unchanged.')
