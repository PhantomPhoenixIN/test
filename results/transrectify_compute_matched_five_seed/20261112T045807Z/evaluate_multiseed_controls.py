#!/usr/bin/env python3
import json, os, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT=Path('/home/swaminathanj/TransRectify_New/content/files/update')
PRED=ROOT/'content/compute_matched_controls/predictions'
OUT=ROOT/'content/compute_matched_controls/results'
SEEDS=(42,7,123,999,2026)
CONTROLS=('mixed','mixed_s3')

# Import the evaluator by a normal, importable module name.  Multiprocessing
# pickles worker functions by module/name; the previous anonymous ``ev``
# import was not registered in child processes and made every pool fail.
sys.path.insert(0, str(ROOT/'content/files'))
import evaluation as ev

def one(path):
    lines=path.read_text().splitlines(); total=0; sums={k:0 for k in ('parsable','compilable','pass1','pass3','pass5')}
    # PBS_NCPUS is not exported on this cluster.  Require an explicit worker
    # count from the submission script so a small allocation cannot
    # accidentally spawn one process per physical core on the whole node.
    workers = int(os.environ.get('EVAL_WORKERS', '1'))
    if workers < 1:
        raise ValueError('EVAL_WORKERS must be at least 1')
    print(f'EVALUATING {path.name} workers={workers}', flush=True)
    with Pool(workers, initializer=ev.init_worker, initargs=(ev.load_tests(str(ROOT/'content/dataset/problem_tests.json')),)) as pool:
        for res in pool.imap(ev.process_instance, lines, chunksize=10):
            if res is None: continue
            total+=1
            for k in sums: sums[k]+=int(res[k])
    return {'total':total,'metrics':{k:(sums[k]/total if total else 0) for k in sums},'prediction_file':str(path)}

def main():
    OUT.mkdir(parents=True,exist_ok=True); results=[]
    for seed in SEEDS:
        for control in CONTROLS:
            path=PRED/f'{control}_seed_{seed}.jsonl'
            if not path.exists(): raise FileNotFoundError(path)
            item=one(path); item.update(seed=seed,control=control); results.append(item)
    summary={}
    for control in CONTROLS:
        rows=[x for x in results if x['control']==control]
        summary[control]={k:{'mean':sum(x['metrics'][k] for x in rows)/len(rows),'values':[x['metrics'][k] for x in rows]} for k in ('pass1','pass3','pass5','parsable','compilable')}
    (OUT/'multiseed_control_results.json').write_text(json.dumps({'runs':results,'summary':summary},indent=2))
    with (OUT/'multiseed_control_results.txt').open('w') as f:
        f.write('Five-seed compute-matched controls\n\n')
        for control in CONTROLS:
            f.write(f'{control}\n')
            for k,v in summary[control].items(): f.write(f'  {k}: mean={v["mean"]:.6f} values={v["values"]}\n')
    print('RUN_STATUS=SUCCESS')
if __name__=='__main__': main()
