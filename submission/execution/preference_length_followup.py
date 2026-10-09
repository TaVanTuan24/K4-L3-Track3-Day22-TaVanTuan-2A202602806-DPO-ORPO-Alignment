"""Recompute NB2 length reports from the original split and saved tokenizer."""
from pathlib import Path
import sys, json, hashlib
import transformers
from datasets import Dataset
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from lab22 import data as D
pref=ROOT/'data/pref'
assert transformers.__version__=="5.17.0"
assert D.split_mismatch(pref,ROOT/'adapters/dpo') is None
fingerprint=D.split_fingerprint(pref)
train=list(Dataset.from_parquet(str(pref/'train.parquet')))
held=list(Dataset.from_parquet(str(pref/'eval.parquet')))
assert len(train)==800 and len(held)==100
D.assert_disjoint(train,held)
model_dir=ROOT/'models/sft-merged'
tok=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True)
count=lambda text:len(tok(text,add_special_tokens=False)["input_ids"])
computed=D.length_stats(train,count=count)
original=json.loads((pref/'stats.json').read_text())
for key in ("n","chosen_median","rejected_median","chosen_longer_frac"):
    assert computed[key]==original[key]
original.update(computed)
(pref/'stats.json').write_text(json.dumps(original,indent=2))
assert D.split_fingerprint(pref)==fingerprint
receipt={"runtime":"Colab CPU", "transformers_version":transformers.__version__, "source":"original NB2 splits and SFT tokenizer; no resplit or retraining", "split_sha256":fingerprint,"tokenizer_sha256":hashlib.sha256((model_dir/'tokenizer.json').read_bytes()).hexdigest(), "train":computed,"eval":D.length_stats(held,count=count),"train_eval_disjoint":True}
(Path(__file__).with_suffix(".json")).write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
