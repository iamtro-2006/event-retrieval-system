"""Evaluate a configured vi->en backend on the small ground-truth set.
Reports corpus BLEU-4 (with add-one smoothing) and character F-score.
Usage: python scripts/evaluate_translation.py --agent vi2en
"""
from __future__ import annotations
import argparse, json, math, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def toks(s): return re.findall(r"\w+|[^\w\s]", s.lower(), flags=re.UNICODE)
def bleu(refs, hyps):
    vals=[]
    for n in range(1,5):
        match=total=0
        for r,h in zip(refs,hyps):
            rc=Counter(tuple(toks(r)[i:i+n]) for i in range(len(toks(r))-n+1))
            hc=Counter(tuple(toks(h)[i:i+n]) for i in range(len(toks(h))-n+1))
            match += sum((rc & hc).values()); total += sum(hc.values())
        vals.append((match+1)/(total+1))
    ref_len=sum(len(toks(x)) for x in refs); hyp_len=sum(len(toks(x)) for x in hyps)
    bp=1 if hyp_len>=ref_len else math.exp(1-ref_len/max(hyp_len,1))
    return 100*bp*math.exp(sum(math.log(x) for x in vals)/4)
def chrf(refs, hyps):
    fs=[]
    for r,h in zip(refs,hyps):
        a=Counter(r.lower().replace(" ","")); b=Counter(h.lower().replace(" ","")); overlap=sum((a&b).values())
        p=overlap/max(sum(b.values()),1); q=overlap/max(sum(a.values()),1)
        fs.append(2*p*q/max(p+q,1e-12))
    return 100*sum(fs)/len(fs)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--agent",default="vi2en"); ap.add_argument("--data",default=str(ROOT/"data/translation/vi_en_ground_truth.jsonl")); args=ap.parse_args()
    from src.translation.factory import get_translator
    rows=[json.loads(x) for x in Path(args.data).read_text(encoding="utf-8").splitlines() if x.strip()]
    # Keep evaluator independent of yaml parser details while using the production factory.
    cfg={"translate_agent":args.agent,"translate":{"vi2en":{}}}
    translator=get_translator(cfg, ROOT)
    hyps=translator.translate_batch([r["vi"] for r in rows])
    refs=[r["en"] for r in rows]
    print(json.dumps({"agent":args.agent,"samples":len(rows),"BLEU":round(bleu(refs,hyps),2),"chrF":round(chrf(refs,hyps),2)},ensure_ascii=False))
if __name__ == "__main__": main()
