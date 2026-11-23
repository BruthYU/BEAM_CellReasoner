"""Freeze C checkpoint 11000 and run CPU acceptance against the original evaluator."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap

import argparse,json,os,shutil,subprocess,tempfile
from pathlib import Path
import numpy as np,torch,yaml
from transformers import AutoTokenizer
from replogle_llm_common import ROOT,read_json,save_json,sha
from replogle_control_common import CellStore
from replogle_control_c import ART,ControlSetDataset,check_inputs
from replogle_control_eval_common import EVAL_ART as AB_ART,verify_inputs as verify_ab,nine_metrics
from replogle_control_c_eval_common import EVAL_ART,FILES,verify_inputs

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-only',action='store_true');args=p.parse_args()
    if args.verify_only:verify_inputs();print('C evaluation inputs verified.');return
    check_inputs();verify_ab();torch.set_num_threads(4)
    assert not (EVAL_ART/'ready_inputs.json').exists(),'Already frozen; do not replace a submitted evaluation plan'
    EVAL_ART.mkdir(parents=True,exist_ok=True)
    record=read_json(ART/'active_submissions.json')[0]
    run=ROOT/'outputs/replogle_llm_control'/ART.name/'n8'/record['run_id']/'formal'
    cfg=read_json(run/'run_config.json');assert cfg['control_cells']==8
    checkpoint=run/'checkpoint-11000';assert read_json(checkpoint/'trainer_state.json')['global_step']==11000
    snapshot=EVAL_ART/'model_checkpoint11000';snapshot.mkdir(exist_ok=True)
    for source in checkpoint.iterdir():
        if source.is_file() and source.name not in ['optimizer.pt','scheduler.pt','trainer_state.json','training_args.bin'] and not source.name.startswith('rng_state_'):
            destination=snapshot/source.name
            try:
                shutil.copy2(source,destination)
            except PermissionError:
                # Active worker checkpoints may be root-readable until its exit
                # cleanup. Copy only; never chmod/chown the training checkpoint.
                subprocess.run(['sudo','-n','cp','--',str(source),str(destination)],check=True)
                subprocess.run(['sudo','-n','chown',f'{os.getuid()}:{os.getgid()}',str(destination)],check=True)
    try:
        source_hash=sha(checkpoint/'model.safetensors')
    except PermissionError:
        source_hash=subprocess.check_output(['sudo','-n','sha256sum',str(checkpoint/'model.safetensors')],text=True).split()[0]
    assert sha(snapshot/'model.safetensors')==source_hash
    for f in AB_ART.glob('*.npy'):shutil.copy2(f,EVAL_ART/f.name)
    plan=read_json(AB_ART/'plan.json')
    plan['models']={'C':{'model':str(snapshot),'selected_checkpoint':str(checkpoint),'model_sha256':sha(snapshot/'model.safetensors'),'selection':'User selected saved checkpoint-11000; validation CE 1.4053879976272583'}}
    plan.update(variant='C',control_cells=8,ab_reference_plan=str(AB_ART/'plan.json'),pds_audit='Exact float32 evaluator pseudobulks, original PDS function, symbol-aligned target exclusion')
    save_json(EVAL_ART/'plan.json',plan)
    store=CellStore();rows=np.load(EVAL_ART/'generation_rows.npy');tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    ds=ControlSetDataset(store,rows,tok,8,seed=420)
    sampled=np.unique(np.linspace(0,len(rows)-1,200,dtype=int))
    for i in sampled:
        row=int(rows[i]);controls=ds.control_indices(row);prefix=ds.inference_prefix(i)
        assert len(controls)==8 and (store.support[controls]).all() and (store.gem[controls]==store.gem[row]).all() and (controls!=row).all()
        assert prefix.count(ds.start)==9 and prefix.count(ds.end)==8
        assert ds[i]['input_ids'][:len(prefix)]==prefix and all(v==-100 for v in ds[i]['labels'][:len(prefix)])
    assert ds.max_length<=cfg['max_sequence_length']
    from replogle_qwen_metrics import evaluate_case
    source=store.sources;config=yaml.safe_load((ROOT/source['vae_run']/'config.yaml').read_text());config['_vae_run']=source['vae_run']
    labels=read_json(ROOT/source['nb_cache']/'labels.json')
    genes=(ROOT/'assets/replogle_scldm2k_gene_order.ensembl.txt').read_text().splitlines();symbols=(ROOT/'assets/replogle_scldm2k_gene_order.symbols.txt').read_text().splitlines()
    targets=sorted(set(symbols)&set(labels['perturbations']))[:2]
    counts=np.random.default_rng(420).poisson(3,(192,2000)).astype(np.float32);counts[64:128,:100]+=4;counts[128:,100:200]+=6
    pert=np.repeat([labels['perturbations'].index(x) for x in ['non-targeting']+targets],64);ct=np.full(192,labels['cell_types'].index('hepg2'))
    fixture=dict(key='synthetic_identity',split='cpu_fixture',cell_line='HepG2',target_cells=128,conditions=2,reference_cells=64)
    out=Path(tempfile.mkdtemp(prefix='cpu_evaluator_',dir=EVAL_ART))
    result=evaluate_case(fixture,out,counts,counts.copy(),pert,ct,labels,genes,np.arange(192),config,smoke=True);v=nine_metrics(result)
    assert all(np.isfinite(x) for x in v.values()) and v['W2']<1e-4 and abs(v['MMD2'])<1e-5 and abs(v['FD'])<1e-4 and v['PDS']==1 and v['P-delta']>.9999
    # Exact PDS means survive saving and reproduce the original fixture result.
    z=np.load(out/'reports/synthetic_identity/eval_results/pds_exact_inputs.npz');assert z['real'].dtype==np.float32
    report=dict(status='passed',device='cpu',control_cells=8,checked_prompts=len(sampled),unique_generation_cells=len(rows),max_sequence_length=ds.max_length,nine_metrics_identity=v,selected_checkpoint=str(checkpoint),gpu_generation='pending worker startup')
    save_json(EVAL_ART/'cpu_preflight.json',report)
    files=[_bootstrap.script_path(f) for f in FILES]+list(snapshot.iterdir())+list(EVAL_ART.glob('*.npy'))+[EVAL_ART/'plan.json',EVAL_ART/'cpu_preflight.json']
    save_json(EVAL_ART/'ready_inputs.json',dict(status='cpu_ready',sha256={str(f.relative_to(ROOT)):sha(f) for f in files if f.is_file()}))
    print(report,flush=True)
if __name__=='__main__':main()
