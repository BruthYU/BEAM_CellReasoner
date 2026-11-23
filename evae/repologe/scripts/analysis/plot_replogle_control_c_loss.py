"""Plot formal C losses across the checkpoint-6000 batch-size transition."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap

import csv,datetime,hashlib,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts/replogle_llm_control/qwen35-08b-fsq-nb-hepg2-c-v1'
OUT=ART/'analysis/loss_curve'

def read_snapshot(run):
    raw=(run/'metrics.jsonl').read_bytes()
    # A live writer may be between the JSON body and its newline.
    complete=raw[:raw.rfind(b'\n')+1]
    return [json.loads(s) for s in complete.splitlines()],hashlib.sha256(complete).hexdigest()

def main():
    record=json.loads((ART/'active_submissions.json').read_text())[0]
    base=ROOT/'outputs/replogle_llm_control'/ART.name/'n8'
    run=base/record['run_id']/'formal'
    cfg=json.loads((run/'run_config.json').read_text())
    boundary=cfg['resumed_step'];old=Path(cfg['source_run'])
    first,old_sha=read_snapshot(old);second,new_sha=read_snapshot(run)
    done=json.loads((run/'completion.json').read_text()) if (run/'completion.json').exists() else None
    restored=None
    if done:
        finals=[i for i,r in enumerate(second) if r['step']==done['steps'] and 'eval_loss' in r]
        if finals:
            restored=second.pop(finals[-1])
    rows=[dict(r,segment='original') for r in first if r['step']<=boundary]
    rows += [dict(r,segment='resumed') for r in second if r['step']>boundary]
    train=sorted([r for r in rows if 'loss' in r],key=lambda r:r['step'])
    validation={}
    for r in rows:
        if 'eval_loss' in r:validation.setdefault(r['step'],r)
    val=[validation[s] for s in sorted(validation)]
    steps=np.array([r['step'] for r in train]);loss=np.array([r['loss'] for r in train])
    assert np.all(np.diff(steps)>0) and np.isfinite(loss).all()
    vs=np.array([r['step'] for r in val]);vl=np.array([r['eval_loss'] for r in val])
    assert np.all(np.diff(vs)>0) and np.isfinite(vl).all()
    starts=np.searchsorted(steps,steps-199);cs=np.r_[0.,loss.cumsum()]
    smooth=(cs[1:]-cs[starts])/(np.arange(len(steps))+1-starts)
    best=min(val,key=lambda r:r['eval_loss']);stamp=datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')
    OUT.mkdir(parents=True,exist_ok=True)
    for filename,header,values in [('C_train.csv',['step','logged_train_ce','trailing_200_step_mean_ce','segment'],zip(steps,loss,smooth,[r['segment'] for r in train])),('C_validation.csv',['step','validation_ce','segment'],[(r['step'],r['eval_loss'],r['segment']) for r in val])]:
        with (OUT/filename).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(header);w.writerows(values)
    summary={'snapshot_time':stamp,'status':'complete' if done else 'in_progress','resume_step':boundary,
        'last_train_step':int(steps[-1]),'last_train_ce':float(loss[-1]),'latest_validation':val[-1],
        'observed_best_validation':best,'best_observation_saved_at_regular_interval':best['step']%cfg['save_steps']==0,
        'validation_cells':cfg['validation_monitor_cells'],'source_run':str(old),'resumed_run':str(run),
        'source_metrics_snapshot_sha256':old_sha,'resumed_metrics_snapshot_sha256':new_sha,
        'excluded':'source rows after resume boundary; smoke updates; final restored-model evaluation',
        'restored_model_evaluation':restored,'total_planned_steps':17418}
    if done:summary['exported_checkpoint']=done['best_checkpoint']
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(14,5.8));fig.subplots_adjust(left=.065,right=.985,bottom=.22,top=.82,wspace=.18)
    for ax in axes:
        ax.plot(steps,loss,lw=.5,alpha=.22,color='#2563a6')
        ax.plot(steps,smooth,lw=1.8,color='#2563a6',label='Train CE: trailing 200-step mean')
        ax.plot(vs,vl,'o--',ms=3,lw=1.6,color='#d26925',label='Validation CE: 6,360 fixed cells')
        ax.axvline(boundary,color='#8055a5',ls=':',lw=1.5,label='Resume at step 6,000')
        ax.scatter([best['step']],[best['eval_loss']],marker='*',s=110,color='#d26925',zorder=5)
        ax.set_xlabel('Optimizer step');ax.set_ylabel('Completion token cross-entropy (nats/token)')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k'));ax.grid(axis='y',alpha=.2)
    axes[0].set_title('Formal training across both configurations',loc='left',fontweight='bold')
    axes[0].set_xlim(0,17800);axes[0].legend(frameon=False,fontsize=9,loc='upper right')
    axes[0].axvline(math.ceil(17418*.03),color='gray',lw=.8,ls=':')
    axes[1].set_title('Zoom: step 1,000 onward',loc='left',fontweight='bold')
    axes[1].set_xlim(1000,17800);axes[1].set_ylim(1.26,1.68)
    saved_note='not saved' if best['step']%cfg['save_steps'] else 'save interval'
    axes[1].annotate(f"Best observed val: {best['eval_loss']:.4f}\nstep {best['step']:,} ({saved_note})",xy=(best['step'],best['eval_loss']),xytext=(10000,1.55),fontsize=9,color='#b65119',arrowprops={'arrowstyle':'->','color':'#b65119'})
    status='Complete' if done else f"In progress: last logged update {steps[-1]:,} / 17,418"
    fig.suptitle('Replogle C | Qwen3.5-0.8B + FSQ-NB | text + 8 control cells\n'+status,fontsize=14,fontweight='bold')
    fig.text(.5,.05,'4 H200 GPUs throughout; microbatch / accumulation: 2 / 16 -> 8 / 4 at step 6,000; global batch stays 128.\n'
        'Only formal updates are shown. Validation uses source-line combination holdouts; no HepG2 selection.\n'+f'Snapshot: {stamp}. Star marks the observed validation minimum; it does not imply a saved checkpoint.',ha='center',fontsize=9,color='#4b5563')
    for ext in ['png','pdf']:fig.savefig(OUT/f'replogle_control_c_loss.{ext}',dpi=200)
    plt.close(fig);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
