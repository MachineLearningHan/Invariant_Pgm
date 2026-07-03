# PAPER: Utility -- verify offline HF cache before qwen_crosswidth.py (Result 7). See RESULTS_TO_CODE.md
"""
Run this FIRST on your server to confirm what's in the local HF cache and how to
address it, before launching qwen_crosswidth.py.

  HF_HOME=/home/models python check_local_cache.py

It prints:
  - which wikitext config directories exist in the cache
  - the exact load_dataset(...) call that will work offline
  - whether Qwen2.5-1.5B / 0.5B weights are present
"""
import os, glob

HF_HOME = os.environ.get("HF_HOME", "/home/models")
HUB = os.path.join(HF_HOME, "hub")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

print(f"HF_HOME = {HF_HOME}")
print(f"hub dir = {HUB}\n")

print("=== datasets cached under hub ===")
ds_dirs = sorted(glob.glob(os.path.join(HUB, "datasets--*")))
for d in ds_dirs:
    print("  ", os.path.basename(d))
wikitext_dirs = [d for d in ds_dirs if "wikitext" in d.lower()]

print("\n=== models cached under hub ===")
for d in sorted(glob.glob(os.path.join(HUB, "models--*"))):
    print("  ", os.path.basename(d))

# also check a possible datasets/ dir layout (older cache)
alt = os.path.join(HF_HOME, "datasets")
if os.path.isdir(alt):
    print(f"\n=== {alt} (alt datasets dir) ===")
    for d in sorted(os.listdir(alt)):
        print("  ", d)

print("\n=== trying to load wikitext offline ===")
loaded = False
try:
    from datasets import load_dataset
    for cfg in ["wikitext-103-raw-v1", "wikitext-2-raw-v1",
                "wikitext-103-v1", "wikitext-2-v1"]:
        try:
            ds = load_dataset("wikitext", cfg, split="train", streaming=True)
            first = next(iter(ds))
            print(f"  OK: load_dataset('wikitext', '{cfg}', split='train')")
            print(f"      first nonempty text sample present: "
                  f"{bool((first.get('text') or '').strip())}")
            print(f"\n  => use:  --dataset wikitext   (config {cfg} resolves)")
            loaded = True
            break
        except Exception as e:
            print(f"  config {cfg}: {type(e).__name__}: {str(e)[:80]}")
except Exception as e:
    print(f"  datasets import/load failed: {e}")

if not loaded:
    print("\n  Could not auto-load wikitext. If a config dir is listed above,")
    print("  point --dataset at its local path, or edit build_probe() to read")
    print("  the raw .parquet/.arrow files directly with datasets.load_from_disk.")

print("\n=== trying to load Qwen tokenizer offline ===")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    print(f"  OK: Qwen2.5-1.5B tokenizer, vocab={tok.vocab_size}")
except Exception as e:
    print(f"  tokenizer load failed: {e}")
    print("  (ensure models--Qwen--Qwen2.5-1.5B is in the hub dir above)")
