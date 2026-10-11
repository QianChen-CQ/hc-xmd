#!/usr/bin/env python3
"""Regenerate all five experimental tables from bundled per-model metrics.
No training, GPU, server access, or historical placeholder rows are used.
"""
from pathlib import Path
import json
import pandas as pd
OUT=Path(__file__).resolve().parents[1]
df=pd.read_csv(OUT/'data/all_audit_rows.csv')
def sci(x,d=2):
    if x==0:return '0'
    mant,e=f'{x:.{d}e}'.split('e');return mant+r'\times10^{'+str(int(e))+'}'
def ms(rows,key,d=4,scale=1,scientific=False):
    a=rows[key].astype(float)*scale;m=a.mean();s=a.std(ddof=1)
    if scientific:return '$'+sci(m)+r'\pm '+sci(s)+'$'
    return f'${m:.{d}f}\\pm {s:.{d}f}$'
def table(caption,label,cols,head,body,note=''):
    return ('\\begin{table*}[!t]\n\\centering\\footnotesize\n\\revcaption{'+caption+'}\n\\label{'+label+'}\n'
      +'\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{'+cols+'}\n\\toprule\n'+head+r' \\'+'\n\\midrule\n'
      +'\n'.join(body)+'\n\\bottomrule\n\\end{tabular}\n'
      + ('\\par\\smallskip\\begin{minipage}{0.98\\textwidth}\\footnotesize '+note+'\\end{minipage}\n' if note else '')+'\\end{table*}\n')

s1=df[(df.phase==1)&(df.audit=='strong')]
body=[]
for group,label in [('RegretNet_3x10','RegretNet (RI)'),('AMenuNet_3x10','AMenuNet (RI)')]:
    r=s1[s1.run_group==group];p=int(r.parameter_count.iloc[0]);menu='Direct map' if r.menu_size.isna().all() else str(int(r.menu_size.iloc[0]))+' + no-sale'
    body.append(' & '.join([label,ms(r,'revenue'),ms(r,'welfare'),ms(r,'regret_mean',scientific=True),ms(r,'pmr',2,100),f'{p:,}',menu])+r' \\')
(OUT/'tables/phase1.tex').write_text(table('Phase I: static $3\\times10$ auctions. Means $\\pm$ sample SD across three new training seeds; regret and PMR use the strong audit.','tab:phase1-main','lccccrl','Method & Revenue & Welfare & Mean regret & PMR (\\%) & Parameters & Representation',body,'RI denotes this review\'s paper-based reimplementation. Direct map: RegretNet computes allocations and payments from bids without a fixed candidate-menu list. AMenuNet selects among 512 learned candidate allocations plus a no-sale option and computes pivot payments. The representation entries describe implemented mechanisms, not missing runs or human-burden scores; configured slots need not be distinct effective outcomes.'))

s2=df[(df.phase==2)&(df.audit=='strong')]
order=[('3x10','RegretNet_PaperReimplementation','RegretNet (RI)'),('3x10','AMenuNet_PaperReimplementation','AMenuNet (RI)'),('3x10','RegretNet_OptimalerBudget','RegretNet (A; budget)'),('3x10','RegretFormer_AuthorsCheckpoint','RegretFormer (A)'),('2x2','AMenuNet_PaperReimplementation','AMenuNet (RI)'),('2x2','RegretNet_OptimalerBudget','RegretNet (A; budget)'),('2x2','RegretFormer_AuthorsCheckpoint','RegretFormer (A)')]
body=[]
for setting,method,label in order:
    r=s2[(s2.method==method)&s2.setting.str.startswith(setting)]
    if setting=='2x2' and method=='AMenuNet_PaperReimplementation':
        body.append(r'\midrule')
    body.append(' & '.join([setting.replace('x',r'$\times$'),label,ms(r,'revenue'),ms(r,'welfare'),ms(r,'regret_mean',scientific=True),'$'+sci(r.regret_max.max())+'$',ms(r,'pmr',2,100)])+r' \\')
(OUT/'tables/phase2.tex').write_text(table('Phase II: static tasks separated by size and implementation source. Means $\\pm$ sample SD over three models; $r_{\\max}^{\\rm all}$ is the largest observed gain across all three models.','tab:phase2main','llccccc','Task & Method/source & Revenue & Welfare & Mean regret & $r_{\\max}^{\\rm all}$ & PMR (\\%)',body,'RI: newly trained paper-based implementation; the $2\\times2$ uniform AMenuNet task is a review-added extension. A: checkpoints released with Optimaler and reevaluated here; these summarize three checkpoint identities, not verified independent training seeds. The two $3\\times10$ RI rows reuse Phase I models and are not additional independent evidence. Training protocols differ across sources; common evaluation does not establish equal-budget architectural superiority. GemNet is disclosed separately in Table~\\ref{tab:gemnet-diagnostic}.'))

g=json.loads((OUT/'audit/audit_phase2.json').read_text())
body=[]
for record in g['gemnet']:
    d=record['transform_diagnostics'];last=record['last_validation']
    fields=[str(record['seed']),str(record['selected']['step']),
        f"{d['before']['revenue_mean']:.0f} / {d['after']['revenue_mean']:.0f}",
        f"{100*sum(d['before_null_choice_rate_by_bidder'])/len(d['before_null_choice_rate_by_bidder']):.0f} / {100*sum(d['after_null_choice_rate_by_bidder'])/len(d['after_null_choice_rate_by_bidder']):.0f}",
        f"{last['revenue_mean']:.6f}",f"{100*last['incompatibility_rate']:.3f}"]
    body.append(' & '.join(fields)+r' \\')
(OUT/'tables/gemnet.tex').write_text(table('GemNet reimplementation diagnostics: selected checkpoints and the separate final-step validation record.','tab:gemnet-diagnostic','rrcccc','Seed & Selected step & Pre/post revenue & Pre/post null (\\%) & Final validation revenue & Incompatible (\\%)',body,'Selected-model pre/post diagnostics use the same 2,048 profiles per seed and both yield zero welfare. All 2,401 finite-grid profiles also yield null allocations. All 49 transformation contexts per seed require no price adjustment; the MILP branch is skipped. Final-step columns refer to the unselected 20,000-step model on 50,000 validation profiles; incompatibility means that any item\'s summed allocation exceeds $1+10^{-6}$. They must not be combined with the selected model\'s audited regret.'))

s3=df[(df.phase==3)&(df.audit=='strong')]
order3=[('Context $3\\times10$','CITransNetNoContext','No-context ablation'),('','CITransNet','CITransNet'),('Preference $2\\times2$','RegretNetPreferenceBaseline','Matched revenue baseline'),('','PreferenceNet','PreferenceNet'),('Fairness $2\\times2$','RegretNetProportionBaseline','Matched revenue baseline'),('','ProportionNet','ProportionNet')]
body=[]
controls={'CITransNet':'CITransNetNoContext','PreferenceNet':'RegretNetPreferenceBaseline','ProportionNet':'RegretNetProportionBaseline'}
for task,m,label in order3:
    r=s3[s3.method==m]
    revenue=ms(r,'revenue')
    if m in controls:
        baseline=s3[s3.method==controls[m]].revenue.mean()
        delta=100*(r.revenue.mean()/baseline-1)
        revenue=r'\makecell{'+revenue+r'\\ \textbf{('+f'{delta:+.2f}'+r'\%)}}'
    body.append(' & '.join([task,label,revenue,ms(r,'welfare'),ms(r,'regret_mean',scientific=True),'$'+sci(r.regret_max.max())+'$',ms(r,'pmr',2,100)])+r' \\')
(OUT/'tables/phase3.tex').write_text(table('Phase III: task-specific comparisons with matched mechanism-update budgets. All three training seeds are retained; strong-audit results are shown.','tab:phase3quant','llccccc',r'Task & Method & \makecell{Revenue /\\change vs. control} & Welfare & Mean regret & $r_{\max}^{\rm all}$ & PMR (\%)',body,'Cells are means $\\pm$ sample SD except $r_{\\max}^{\\rm all}$, which is the observed maximum across all seeds. Bold percentages below method revenues are $100(\\bar R_{\\rm method}/\\bar R_{\\rm control}-1)$, using unrounded three-seed means and the preceding control row; they are not averages of seedwise ratios. The no-context control is CITransNet with public-context inputs masked in training and evaluation. Its seed 0 collapses to near-zero allocation (revenue $5.19\\times10^{-12}$) and remains in the aggregate; +56.19\\% therefore describes this three-seed mean contrast, not a stable causal context benefit. PreferenceNet adds classifier computation and different batch ordering. Context-dependent valuations differ from the static uniform tasks.'))
body=[]
for m,label in [('RegretNetPreferenceBaseline','Preference revenue baseline'),('PreferenceNet','PreferenceNet'),('RegretNetProportionBaseline','Fairness revenue baseline'),('ProportionNet','ProportionNet')]:
    r=s3[s3.method==m]
    body.append(' & '.join([label,ms(r,'absolute_tvf_sum',5),ms(r,'paper_unfairness_sum',5),ms(r,'pca',2,100),ms(r,'tvf_constraint_satisfaction_1e3',2,100)])+r' \\')
(OUT/'tables/targets.tex').write_text(table('Task-specific allocation metrics for the two $2\\times2$ comparisons. These are synthetic criteria, not human-subject measurements.','tab:task-targets','lcccc','Method & Absolute TVF & Positive-part unfairness & Synthetic pass (\\%) & Strict TVF pass (\\%)',body,'Means $\\pm$ sample SD across three seeds. Lower is better for the two unnormalized unfairness sums. Synthetic pass uses the same fixed reference threshold in both comparisons; strict TVF pass uses a maximum allocation-distance threshold of $10^{-3}$. These two pass rates test different conditions and need not agree.'))

# Main-text result values use the same unrounded records as the tables.
# This file contains display values only; raw data remain unchanged.
numbers={}
def number(name,value,d=4,scientific=False,integer=False):
    numbers['HC'+name]=f'{int(value):,}' if integer else sci(value,d) if scientific else f'{value:.{d}f}'
def avg(r,key): return float(r[key].mean())
def change(method,control,key): return 100*(avg(method,key)/avg(control,key)-1)
rn=s1[s1.run_group=='RegretNet_3x10']
am=s1[s1.run_group=='AMenuNet_3x10']
for label,r in [('RegretNet',rn),('AMenuNet',am)]:
    for metric in ['revenue','welfare']:
        number('Static'+label+metric.title(),avg(r,metric))
    number(label+'Parameters',r.parameter_count.iloc[0],integer=True)
number('AMenuRevenueGain',change(am,rn,'revenue'),2)
number('AMenuWelfareLoss',-change(am,rn,'welfare'),2)
for size,word in [('3x10','Large'),('2x2','Small')]:
    for method,label in [('RegretNet_OptimalerBudget','RegretNet'),('RegretFormer_AuthorsCheckpoint','RegretFormer')]:
        r=s2[(s2.method==method)&s2.setting.str.startswith(size)]
        number(word+label+'Revenue',avg(r,'revenue'))
        number(word+label+'Regret',avg(r,'regret_mean'),2,scientific=True)
rr=df[(df.phase==1)&(df.run_group=='RegretNet_3x10')&(df.audit=='random128')]
if rr.empty:
    candidates=df[(df.phase==1)&(df.run_group=='RegretNet_3x10')]
    audits=[a for a in candidates.audit.unique() if '128' in a]
    if len(audits)!=1: raise ValueError('Expected one random-128 audit')
    rr=candidates[candidates.audit==audits[0]]
number('RandomRegret',avg(rr,'regret_mean'),2,scientific=True)
number('StrongRegret',avg(rn,'regret_mean'),2,scientific=True)
number('RandomPMR',100*avg(rr,'pmr'),3)
number('StrongPMR',100*avg(rn,'pmr'),3)
ci=s3[s3.method=='CITransNet'];nc=s3[s3.method=='CITransNetNoContext']
for label,r in [('Context',ci),('NoContext',nc)]:
    number(label+'Revenue',avg(r,'revenue'))
    number(label+'RevenueSD',float(r.revenue.std(ddof=1)))
number('ContextRevenueChange',change(ci,nc,'revenue'),2)
for seed,word in [(0,'Zero'),(1,'One'),(2,'Two')]:
    a=float(nc[nc.seed==seed].revenue.iloc[0])
    number('NoContextSeed'+word,a,2 if seed==0 else 4,scientific=(seed==0))
    if seed:
        b=float(ci[ci.seed==seed].revenue.iloc[0])
        number('ContextSeed'+word+'Gain',100*(b/a-1),2)
pref=s3[s3.method=='PreferenceNet'];prbase=s3[s3.method=='RegretNetPreferenceBaseline']
prop=s3[s3.method=='ProportionNet'];fbase=s3[s3.method=='RegretNetProportionBaseline']
number('PreferencePassControl',100*avg(prbase,'pca'),2)
number('PreferencePassMethod',100*avg(pref,'pca'),2)
number('PreferenceFairnessReduction',-change(pref,prbase,'absolute_tvf_sum'),2)
number('PreferenceRevenueLoss',-change(pref,prbase,'revenue'),2)
number('PreferenceStrictPass',100*avg(pref,'tvf_constraint_satisfaction_1e3'),2)
number('ProportionFairnessReduction',-change(prop,fbase,'paper_unfairness_sum'),2)
number('ProportionRevenueLoss',-change(prop,fbase,'revenue'),2)
number('ProportionRegretIncrease',change(prop,fbase,'regret_mean'),2)
last_rates=[100*r['last_validation']['incompatibility_rate'] for r in g['gemnet']]
number('GemIncompatLow',min(last_rates),3)
number('GemIncompatHigh',max(last_rates),3)
(OUT/'tables/numbers.tex').write_text('% Generated from data/all_audit_rows.csv and audit/audit_phase2.json.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in numbers.items()))
(OUT/'data/display_numbers.json').write_text(json.dumps(numbers,indent=2)+'\n')

