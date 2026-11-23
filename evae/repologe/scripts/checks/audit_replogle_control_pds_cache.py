"""Recover PDS from saved cell expressions, preserving the evaluator's float32 means."""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap

import argparse,json,hashlib
from pathlib import Path
import draccus,h5py,numpy as np,torch
from sc_evae.config.training import TrainConfig
from sc_evae.training.train_utils import to_metric_space
from sc_evae.metrics.state_metrics import discrimination_score_l1,aggregate_per_pert
from replogle_llm_common import ROOT,read_json,save_json,sha
from replogle_control_eval_common import EVAL_ART,nine_metrics

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variant',choices=['A','B'],required=True);args=p.parse_args()
 record=next(r for r in read_json(EVAL_ART/'active_submissions.json') if r['variant']==args.variant)
 output=Path(record['output']);case=next(c for c in read_json(EVAL_ART/'plan.json')['cases'] if c['key']=='hepg2_all__HepG2')
 report=output/'reports'/case['key'];dest=report/'eval_results'
 with (report/'eval_bridge/config.yaml').open() as f: cfg=draccus.load(TrainConfig,f)
 source=read_json(EVAL_ART.parent/'sources.json');labels=read_json(ROOT/source['nb_cache']/'labels.json')['perturbations']
 genes=(ROOT/'assets/replogle_scldm2k_gene_order.ensembl.txt').read_text().splitlines();symbols=(ROOT/'assets/replogle_scldm2k_gene_order.symbols.txt').read_text().splitlines()
 wanted=np.sort(np.concatenate([np.load(EVAL_ART/f"{case['key']}_{k}.npy") for k in ['targets','controls']]))
 torch.set_num_threads(16)
 with h5py.File(output/'generated_cells.h5','r') as h:
  assert h.attrs['complete'];rows=h['global_rows'][:];selected=np.searchsorted(rows,wanted);np.testing.assert_array_equal(rows[selected],wanted)
  pert=h['pert_idx'][:][selected];real=np.empty((len(selected),2000),np.float32);pred=np.empty_like(real)
  # Same selected order, batch size, conversion function and dtype as scripts/eval.py.
  for start in range(0,len(selected),256):
   ix=selected[start:start+256];x=torch.from_numpy(h['true_counts'][ix]);y=torch.from_numpy(h['predicted_counts'][ix]);lib=x.sum(-1)
   real[start:start+len(ix)]=to_metric_space(x,cfg.dataset,library_size=lib,hvg_library=lib[:,None],basis='hvg_row_sum').numpy()
   pred[start:start+len(ix)]=to_metric_space(y,cfg.dataset,library_size=lib,hvg_library=lib[:,None],basis='hvg_row_sum').numpy()
 order=sorted((labels[int(k)],int(k)) for k in np.unique(pert) if labels[int(k)]!='non-targeting');names=[n for n,k in order]
 assert len(names)==case['conditions'] and len(pert)-int((pert==labels.index('non-targeting')).sum())==case['target_cells']
 rp=np.stack([real[pert==k].mean(0) for n,k in order]);pp=np.stack([pred[pert==k].mean(0) for n,k in order]);cp=real[pert==labels.index('non-targeting')].mean(0)
 # DE code explicitly casts cell arrays to float64 before means. Its saved means
 # thus need not reproduce ranks from the evaluator's float32 pseudobulks.
 rd=np.stack([np.log1p(np.expm1(real[pert==k].astype(np.float64).mean(0))) for n,k in order]);pd=np.stack([np.log1p(np.expm1(pred[pert==k].astype(np.float64).mean(0))) for n,k in order]);cd=np.log1p(np.expm1(real[pert==labels.index('non-targeting')].astype(np.float64).mean(0)))
 del real,pred
 kw=dict(metric='l1',exclude_target_gene=True)
 original=discrimination_score_l1(rp,pp,cp,names,gene_names=genes,**kw)
 saved=read_json(dest/'state_per_pert.json');errors={n:abs(original[n]-saved[n]['discrimination_score_l1']) for n in names};assert max(errors.values())<1e-10, max(errors.values())
 print(args.variant,'exact cached-cell PDS matches original for all',len(names),'perturbations',flush=True)
 recovered_de=discrimination_score_l1(rd,pd,cd,names,gene_names=genes,**kw)
 changed={n:{'float32_original':original[n],'float64_de_roundtrip':recovered_de[n],'rank_change':round((recovered_de[n]-original[n])*len(names))} for n in names if recovered_de[n]!=original[n]}
 corrected=discrimination_score_l1(rp,pp,cp,names,gene_names=symbols,**kw)
 audit={'status':'passed','method':'Original PDS function on exact float32 pseudobulks rebuilt from cached cell expressions; symbol-aligned target exclusion',
  'original':aggregate_per_pert(original),'symbol_aligned_target_excluded':aggregate_per_pert(corrected),'original_pds_cache_max_error':max(errors.values()),
  'de_float64_roundtrip_changed_conditions':changed,'de_float64_roundtrip_max_error':max(abs(recovered_de[n]-original[n]) for n in names),
  'real_pseudobulk_dtype_max_difference':float(np.max(np.abs(rd-rp))),'pred_pseudobulk_dtype_max_difference':float(np.max(np.abs(pd-pp))),
  'heldout_target_genes_present_in_panel':len(set(names)&set(symbols)),'per_pert_symbol_aligned_pds':corrected,'script_sha256':sha(Path(__file__)),
  'source_cache':str(output/'generated_cells.h5'),'source_summary_sha256':sha(dest/'summary_metrics.json')}
 if (dest/'summary_metrics_symbol_aligned.json').exists():
  previous=read_json(dest/'pds_target_gene_audit.json')['per_pert_symbol_aligned_pds']
  audit['previous_corrected_max_difference']=max(abs(previous[n]-corrected[n]) for n in names)
  assert audit['previous_corrected_max_difference']<1e-10,'Existing corrected result differs; review before publishing'
 save_json(dest/'pds_cached_cells_audit.json',audit)
 if not (report/'completion.json').exists():
  save_json(dest/'pds_target_gene_audit.json',audit)
  summary=read_json(dest/'summary_metrics.json');summary['cell_eval']['discrimination_score_l1']=audit['symbol_aligned_target_excluded'];save_json(dest/'summary_metrics_symbol_aligned.json',summary)
  per=read_json(dest/'per_pert_metrics.json');assert len(per)==case['conditions'] and sum(x['n_cells'] for x in per.values())==case['target_cells']
  done={'status':'complete',**case,'nine_metrics':nine_metrics(summary),'eval_code_sha256':sha(ROOT/'scripts/eval.py'),'no_gene_or_cell_line_pooling':True,'recovery':'CPU-only PDS audit from cached expressions; no regeneration or DE recalculation'}
  save_json(report/'completion.json',done)
  reports=read_json(output/'reports_progress.json');assert not any(r['key']==case['key'] for r in reports);reports.append({**case,'nine_metrics':nine_metrics(summary)});save_json(output/'reports_progress.json',reports)
  assert len(reports)==8
  aggregates={}
  for split in ['combination_validation','combination_test']:
   group=[r for r in reports if r['split']==split];aggregates[split]={'aggregation':'unweighted cell-line mean; per-line scores retained','nine_metrics':{m:float(np.mean([r['nine_metrics'][m] for r in group])) for m in group[0]['nine_metrics']}}
  save_json(output/'all_metrics.json',{'variant':args.variant,'reports':reports,'combination_macro_cell_line':aggregates,'protocol':read_json(EVAL_ART/'plan.json')})
  save_json(output/'completion.json',{'status':'complete','variant':args.variant,'reports':8,'all_nine_metrics_reported':True,'source_models_unchanged':True,'recovered_on_cpu':True,'original_rjob_status':'Failed at DE-pseudobulk PDS roundtrip assertion; original logs retained','recovery_script':str(Path(__file__).resolve())})
 print(json.dumps({k:v for k,v in audit.items() if k!='per_pert_symbol_aligned_pds'},indent=2),flush=True)
if __name__=='__main__':main()
