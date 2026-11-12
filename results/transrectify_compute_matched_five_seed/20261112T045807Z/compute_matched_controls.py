#!/usr/bin/env python3
"""Compute-matched controls for the CodeT5 curriculum experiment.

This run trains two fresh models from CodeT5-base for the same fixed number
of optimizer updates used as the curriculum budget: (1) mixed S1+S2+S3 and
(2) S3-only.  It then generates the standard greedy + five sampled outputs
and evaluates them with the existing execution-based evaluator.
"""
import argparse, contextlib, gc, json, logging, math, os, random, re, shutil, socket, subprocess, sys, time, traceback
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from accelerate import Accelerator

ROOT = Path("/home/swaminathanj/TransRectify_New/content/files/update")
DATASET = Path("/home/swaminathanj/TransRectify_New/content/datasets/codenet_combined_translator_dataset_astfixed.jsonl")
BASE_MODEL = Path("/home/swaminathanj/CodeFixNet/content/models/CodeT5/CodeT5")
RUN_ROOT = ROOT / "content" / "compute_matched_controls"
MODEL_ROOT = RUN_ROOT / "models"
PRED_ROOT = RUN_ROOT / "predictions"
RESULT_ROOT = RUN_ROOT / "results"
LOG_ROOT = RUN_ROOT / "logs"

PROMPT = """### Task:
Translate the following {input_language} program into Java.
Preserve semantics and structure.
Output only valid Java source code.

### Source Code:
{source_code}

### Java Code:
"""

def setup_logging(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("compute_matched_controls")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); log.addHandler(sh)
    fh = logging.FileHandler(path); fh.setFormatter(fmt); log.addHandler(fh)
    return log

def reference(obj):
    return {"StarCoder": "translated_java_code", "QwenCoder": "qwencoder_translation_clean", "DeepSeekCoder": "deepseekcoder_translation_clean"}.get(obj.get("teacher")) and obj.get({"StarCoder": "translated_java_code", "QwenCoder": "qwencoder_translation_clean", "DeepSeekCoder": "deepseekcoder_translation_clean"}[obj.get("teacher")])

def load_records(score, split, scores, seed):
    rows = []
    with DATASET.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            if obj.get("split") != split or obj.get("output_language") != "Java": continue
            if score is not None and obj.get("score") != score: continue
            if scores is not None and obj.get("score") not in scores: continue
            src, tgt = obj.get("source_code"), reference(obj)
            if isinstance(src, str) and isinstance(tgt, str) and src.strip() and tgt.strip():
                rows.append({"input_language": obj.get("input_language"), "source": src.strip(), "target": tgt.strip()})
    random.Random(seed).shuffle(rows)
    return rows

class TranslationDataset(Dataset):
    def __init__(self, rows, tok): self.rows, self.tok = rows, tok
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]; p = PROMPT.format(input_language=r["input_language"], source_code=r["source"])
        x = self.tok(p, truncation=True, max_length=512)
        y = self.tok(r["target"], truncation=True, max_length=512)["input_ids"]
        return {"input_ids": x["input_ids"], "attention_mask": x["attention_mask"], "labels": y or [self.tok.eos_token_id]}

def collate(batch, pad):
    mi, ml = max(len(x["input_ids"]) for x in batch), max(len(x["labels"]) for x in batch)
    return {"input_ids": torch.tensor([x["input_ids"]+[pad]*(mi-len(x["input_ids"])) for x in batch]),
            "attention_mask": torch.tensor([x["attention_mask"]+[0]*(mi-len(x["attention_mask"])) for x in batch]),
            "labels": torch.tensor([[t if t != pad else -100 for t in x["labels"]+[pad]*(ml-len(x["labels"]))] for x in batch])}

def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)

def train(strategy, target_updates, seed, log, output_strategy=None, model_dir=None):
    output_strategy = output_strategy or strategy
    out = MODEL_ROOT / output_strategy / "best"
    marker = out / "COMPLETED.json"
    if marker.exists():
        log.info("%s already completed; reusing validated checkpoint", strategy)
        return json.loads(marker.read_text())
    random.seed(seed); torch.manual_seed(seed)
    scores = [1, 2, 3] if strategy == "mixed" else None
    score = None if strategy == "mixed" else 3
    accelerator = Accelerator(mixed_precision="fp16")
    model_dir = model_dir or BASE_MODEL
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    if tok.pad_token is None: tok.add_special_tokens({"pad_token": tok.eos_token})
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir)); model.resize_token_embeddings(len(tok))
    model.gradient_checkpointing_enable(); model.config.use_cache = False
    train_rows = load_records(score, "train", scores, seed)
    val_rows = load_records(score, "val", scores, seed + 1)
    train_dl = DataLoader(TranslationDataset(train_rows, tok), batch_size=8, shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id))
    val_dl = DataLoader(TranslationDataset(val_rows, tok), batch_size=8, shuffle=False, collate_fn=lambda b: collate(b, tok.pad_token_id))
    opt = AdamW(model.parameters(), lr=5e-5)
    model, opt, train_dl, val_dl = accelerator.prepare(model, opt, train_dl, val_dl)
    sched = get_linear_schedule_with_warmup(opt, 100, target_updates)
    best, best_step, updates, epochs = float("inf"), 0, 0, 0
    while updates < target_updates:
        epochs += 1; model.train(); total = 0.0; batches = 0
        for batch in train_dl:
            if updates >= target_updates: break
            loss = model(**batch).loss; accelerator.backward(loss); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            updates += 1; batches += 1; total += float(loss.detach())
        model.eval(); vloss = 0.0; n = 0
        with torch.no_grad():
            for batch in val_dl: vloss += float(model(**batch).loss); n += 1
        log.info("strategy=%s epoch=%d updates=%d/%d train_loss=%.6f val_loss=%.6f", strategy, epochs, updates, target_updates, total/max(batches,1), vloss/max(n,1))
        if vloss/max(n,1) < best:
            best, best_step = vloss/max(n,1), updates
            out.mkdir(parents=True, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(str(out)); tok.save_pretrained(str(out))
    meta = {"strategy": strategy, "output_strategy": output_strategy, "seed": seed, "target_updates": target_updates, "actual_updates": updates, "epochs": epochs, "best_val_loss": best, "best_update": best_step, "train_examples": len(train_rows), "val_examples": len(val_rows), "base_model": str(model_dir), "dataset": str(DATASET)}
    atomic_json(marker, meta); del model, tok; gc.collect(); torch.cuda.empty_cache(); return meta

def extract_class(text):
    if not text: return None
    text = re.sub(r"```java|```", "", text, flags=re.I).strip(); lines=text.splitlines(); start=None
    for i,line in enumerate(lines):
        if "class " in line: start=i; break
    if start is None: return None
    out=[]; brace=0
    for line in lines[start:]:
        out.append(line); brace += line.count("{")-line.count("}")
        if brace == 0: break
    code="\n".join(out).strip()
    return "import java.io.*;\nimport java.util.*;\n\n"+code if "class " in code and code.count("{")==code.count("}") else None

def generate(strategy, log, model_strategy=None):
    model_strategy = model_strategy or strategy
    pred = PRED_ROOT / f"{strategy}.jsonl"
    if pred.exists() and pred.stat().st_size > 0: log.info("reusing predictions %s", pred); return pred
    import importlib.util
    spec=importlib.util.spec_from_file_location("pk", str(ROOT/"content/files/pass_at_k_predictions.py")); pk=importlib.util.module_from_spec(spec); spec.loader.exec_module(pk)
    pk.DATASET_PATH=str(DATASET); rows=pk.load_test_samples_dedup(); model_path=str(MODEL_ROOT/model_strategy/"best")
    model,tok,typ=pk.load_model(model_path); pred.parent.mkdir(parents=True, exist_ok=True)
    with pred.open("w", encoding="utf-8") as f:
        for start in range(0,len(rows),16):
            batch=rows[start:start+16]; prompts=[PROMPT.format(input_language=x["input_language"],source_code=x["source_code"]) for x in batch]
            enc=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=512).to(pk.DEVICE)
            greedy=pk.generate_batch(model,tok,typ,enc,False); sampled=pk.generate_batch(model,tok,typ,enc,True)
            for i,s in enumerate(batch):
                rec={"problem_id":s["problem_id"],"input_language":s["input_language"],"greedy":extract_class(greedy[i])}
                for k in range(5): rec[f"candidate_{k+1}"]=extract_class(sampled[i*5+k])
                f.write(json.dumps(rec)+"\n")
            f.flush(); log.info("strategy=%s generated=%d/%d",strategy,min(start+len(batch),len(rows)),len(rows))
    del model,tok; gc.collect(); torch.cuda.empty_cache(); return pred

def evaluate(pred, strategy, log):
    import importlib.util, io
    spec=importlib.util.spec_from_file_location("ev", str(ROOT/"content/files/evaluation.py")); ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    ev.PRED_FILE=str(pred); ev.TEST_FILE=str(ROOT/"content/dataset/problem_tests.json")
    ev.TESTS=ev.load_tests(ev.TEST_FILE); lines=pred.read_text().splitlines(); stats=defaultdict(lambda: {k:0 for k in ("total","parsable","compilable","pass1","pass3","pass5")})
    from multiprocessing import Pool
    total=0; sums={k:0 for k in ("parsable","compilable","pass1","pass3","pass5")}
    workers = int(os.environ.get("PBS_NCPUS", os.cpu_count() or 1))
    log.info("evaluation_workers=%d", workers)
    with Pool(workers, initializer=ev.init_worker, initargs=(ev.TESTS,)) as pool:
        for res in pool.imap(ev.process_instance, lines, chunksize=10):
            if res is None: continue
            total += 1
            for k in sums: sums[k] += int(res[k])
            s=stats[res["language"]]; s["total"] += 1
            for k in sums: s[k] += int(res[k])
    result={"strategy":strategy,"prediction_file":str(pred),"total":total,"metrics":{k:sums[k]/total if total else 0 for k in sums},"language_metrics":{lang:{k:(v[k]/v["total"] if v["total"] else 0) for k in sums} for lang,v in stats.items()}}
    atomic_json(RESULT_ROOT/f"{strategy}.json", result); log.info("RESULT %s", json.dumps(result["metrics"], sort_keys=True)); return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--target-updates",type=int,default=18000); ap.add_argument("--early-updates",type=int,default=None); ap.add_argument("--s3-updates",type=int,default=None); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--run-id",default=None); ap.add_argument("--strategy",choices=["mixed","s3_matched","mixed_s3"],default=None); ap.add_argument("--skip-train",action="store_true"); ap.add_argument("--skip-generate",action="store_true"); ap.add_argument("--skip-eval",action="store_true"); args=ap.parse_args()
    LOG_ROOT.mkdir(parents=True,exist_ok=True); RESULT_ROOT.mkdir(parents=True,exist_ok=True)
    log=setup_logging(LOG_ROOT/"compute_matched_controls.log")
    log.info("START host=%s python=%s torch=%s cuda=%s device_count=%s target_updates=%d seed=%d",socket.gethostname(),sys.executable,torch.__version__,torch.cuda.is_available(),torch.cuda.device_count(),args.target_updates,args.seed)
    log.info("DATASET=%s BASE_MODEL=%s",DATASET,BASE_MODEL)
    run_id = args.run_id or (args.strategy or "controls")
    strategies = (run_id,) if args.strategy else ("mixed", "s3_matched")
    if not args.skip_train:
        if args.strategy == "mixed_s3":
            early = args.early_updates if args.early_updates is not None else args.target_updates
            s3 = args.s3_updates if args.s3_updates is not None else 0
            phase = f"{run_id}_mixed_phase"
            train("mixed", early, args.seed, log, output_strategy=phase)
            train("s3_matched", s3, args.seed, log, output_strategy=run_id, model_dir=MODEL_ROOT/phase/"best")
        else:
            for strategy in strategies: train(args.strategy or strategy,args.target_updates,args.seed,log,output_strategy=run_id)
    if args.skip_eval:
        if not args.skip_generate:
            for strategy in strategies: generate(strategy,log,model_strategy=run_id)
        log.info("RUN_STATUS=SUCCESS predictions_complete evaluation_deferred")
        return
    results=[]
    for strategy in strategies:
        pred=(PRED_ROOT / f"{strategy}.jsonl") if args.skip_generate else generate(strategy,log,model_strategy=run_id)
        results.append(evaluate(pred,strategy,log))
    report=RESULT_ROOT/"compute_matched_controls.txt"
    lines=["Compute-matched curriculum controls", "", f"target optimizer updates: {args.target_updates}", f"seed: {args.seed}", "", "Results:"]
    for r in results: lines.append(f"{r['strategy']}: total={r['total']} Pass@1={r['metrics']['pass1']:.6f} Pass@3={r['metrics']['pass3']:.6f} Pass@5={r['metrics']['pass5']:.6f} parsable={r['metrics']['parsable']:.6f} compilable={r['metrics']['compilable']:.6f}")
    lines += ["", "Interpretation: compare these controls with the manuscript's seed-42 curriculum result. If the curriculum remains higher than both controls at the same update budget, the result supports an ordering effect beyond additional training exposure. If the controls close the gap, the earlier improvement is more plausibly attributable to additional updates or repeated exposure.", "", f"Machine-readable results: {RESULT_ROOT}"]
    report.write_text("\n".join(lines)+"\n"); log.info("RUN_STATUS=SUCCESS report=%s",report)
if __name__ == "__main__": main()
