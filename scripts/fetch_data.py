#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Download and clean real training data for Persian, English and Python.

Everything comes from public HuggingFace parquet endpoints (no auth, no
`datasets` dependency - only `pyarrow`).  The cleaning pipeline is the part that
matters at small scale:

  * per-document script check      a "Persian" document that is 40% Latin is not
                                   Persian data, it is noise
  * exact and near duplicate       hash of the normalised text + shingle
    removal                        signature (MinHash-lite)
  * paragraph level dedup          kills Wikipedia/boilerplate repetition
  * length and quality gates       too short, too repetitive, too many symbols
  * Python: real AST validation    `ast.parse` is a far stronger filter than any
                                   regex, plus docstring/comment ratio gates
  * bridge data generation         real (docstring -> code) pairs mined from the
                                   downloaded Python files, in Persian *and*
                                   English instruction form

Usage
    python3 scripts/fetch_data.py --all --budget-mb 900
    python3 scripts/fetch_data.py --fa --budget-mb 600          # Persian only
    python3 scripts/fetch_data.py --list                        # show sources
"""
import argparse
import ast
import hashlib
import io
import json
import os
import random
import re
import sys
import unicodedata
import urllib.request

HF = "https://huggingface.co/api/datasets"

# ----------------------------------------------------------------------------
# Source registry.  `weight` is the default share of the byte budget.
# Check the licence of every dataset before shipping a model trained on it.
SOURCES = {
    "fa": [
        # Persian Wikipedia: clean, encyclopedic, CC BY-SA 3.0 / GFDL.
        ("wikimedia/wikipedia", "20231101.fa", "text", 1.0),
    ],
    "en": [
        ("wikimedia/wikipedia", "20231101.en", "text", 1.0),
    ],
    "py": [
        # Small, already filtered code corpora (permissive licences only).
        ("bigcode/the-stack-smol", "data/python", "content", 1.0),
        ("codeparrot/codeparrot-clean-valid", "default", "content", 1.0),
    ],
}

ZWNJ = "\u200c"


def log(msg):
    print(msg, flush=True)


def http_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def list_shards(dataset, config):
    """Returns the parquet URLs of a dataset config (train split)."""
    for cfg in (config, "default", "train"):
        url = f"{HF}/{dataset}/parquet/{cfg}/train"
        try:
            data = http_json(url)
        except Exception:
            continue
        if isinstance(data, list) and data:
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list) and v:
                    return v
    # some datasets expose <config>/<split> directly
    try:
        data = http_json(f"{HF}/{dataset}/parquet")
        if isinstance(data, dict):
            for k, v in data.items():
                if config in k and isinstance(v, dict):
                    for vv in v.values():
                        if isinstance(vv, list):
                            return vv
    except Exception:
        pass
    return []


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return os.path.getsize(path)
    tmp = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "slm-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            if total % (32 << 20) == 0:
                log(f"      {total/1048576:.0f} MB")
    os.replace(tmp, path)
    return os.path.getsize(path)


# ------------------------------------------------------------------ cleaning
def script_ratios(text):
    arabic = latin = digits = other = 0
    for ch in text[:4000]:
        o = ord(ch)
        if 0x0600 <= o <= 0x06FF or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF:
            arabic += 1
        elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            latin += 1
        elif ch.isdigit():
            digits += 1
        elif not ch.isspace():
            other += 1
    n = max(1, arabic + latin + digits + other)
    return arabic / n, latin / n, other / n


WIKI_JUNK = re.compile(r"(?m)^\s*(==+.*==+|\*|\||\{\||\}|!)\s*$")
MULTI_NL = re.compile(r"\n{3,}")
SPACES = re.compile(r"[ \t]{3,}")
REF = re.compile(r"\[\d+\]|\[\[|\]\]|\{\{[^}]*\}\}")


def clean_prose(text):
    text = unicodedata.normalize("NFC", text)
    text = REF.sub(" ", text)
    text = WIKI_JUNK.sub("", text)
    text = SPACES.sub("  ", text)
    text = MULTI_NL.sub("\n\n", text)
    return text.strip()


def shingles(text, k=5, keep=12):
    words = text.split()
    if len(words) < k:
        return frozenset()
    hs = []
    for i in range(0, len(words) - k + 1, 2):
        h = hashlib.blake2b(" ".join(words[i:i + k]).encode(), digest_size=8).digest()
        hs.append(int.from_bytes(h, "little"))
    hs.sort()
    return frozenset(hs[:keep])


class Dedup:
    """Exact hashes + MinHash-lite near duplicate detection."""

    def __init__(self, jaccard=0.75):
        self.exact = set()
        self.buckets = {}
        self.jaccard = jaccard
        self.dropped_exact = 0
        self.dropped_near = 0

    def accept(self, text):
        h = hashlib.blake2b(text.encode(), digest_size=16).digest()
        if h in self.exact:
            self.dropped_exact += 1
            return False
        sig = shingles(text)
        if sig:
            keys = sorted(sig)[:3]
            cands = set()
            for k in keys:
                cands |= self.buckets.get(k, set())
            for other in list(cands)[:24]:
                inter = len(sig & other)
                if inter / max(1, len(sig | other)) >= self.jaccard:
                    self.dropped_near += 1
                    return False
            for k in keys:
                self.buckets.setdefault(k, set()).add(sig)
        self.exact.add(h)
        return True


def keep_prose(text, lang):
    if len(text) < 300:
        return False
    ar, la, other = script_ratios(text)
    if lang == "fa" and (ar < 0.55 or la > 0.25):
        return False
    if lang == "en" and (la < 0.60 or ar > 0.05):
        return False
    if other > 0.25:
        return False
    lines = text.split("\n")
    if len(lines) > 3:
        uniq = len(set(lines)) / len(lines)
        if uniq < 0.5:  # repetitive boilerplate
            return False
    return True


# -------------------------------------------------------------------- python
def python_quality(src):
    """Returns (ok, comment_ratio, n_defs) using a real AST parse."""
    if not (200 < len(src) < 20000):
        return False, 0.0, 0
    if "\t" in src and "    " in src:
        return False, 0.0, 0
    lines = src.split("\n")
    if any(len(l) > 300 for l in lines):
        return False, 0.0, 0
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False, 0.0, 0
    defs = sum(1 for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
    if defs == 0:
        return False, 0.0, 0
    comments = sum(1 for l in lines if l.strip().startswith("#"))
    docstrings = sum(
        1 for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module))
        and ast.get_docstring(n))
    ratio = (comments + docstrings) / max(1, len(lines))
    if ratio > 0.6:  # mostly prose, not code
        return False, ratio, defs
    return True, ratio, defs


FA_ASK = [
    "یک تابع پایتون بنویس که {}",
    "کد پایتون برای این کار بنویس: {}",
    "با پایتون تابعی بنویس که {}",
    "چطور در پایتون {}؟ کد بنویس",
]
EN_ASK = [
    "write a python function that {}",
    "show me python code that {}",
    "implement a python helper that {}",
]


def mine_bridge(src, rng, out):
    """Extracts (docstring -> code) pairs: real instruction data from real code."""
    try:
        tree = ast.parse(src)
    except Exception:
        return 0
    lines = src.split("\n")
    made = 0
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        doc = ast.get_docstring(node)
        if not doc:
            continue
        first = doc.strip().split("\n")[0].strip().rstrip(".")
        if not (12 <= len(first) <= 140) or "\t" in first:
            continue
        end = getattr(node, "end_lineno", None)
        if not end or end - node.lineno > 60:
            continue
        code = "\n".join(lines[node.lineno - 1:end]).rstrip()
        if len(code) < 60:
            continue
        desc = first[0].lower() + first[1:]
        q = rng.choice(FA_ASK).format(desc) if rng.random() < 0.5 else rng.choice(EN_ASK).format(desc)
        out.write(f"<|user|>{q}<|assistant|>```python\n{code}\n```<|endoftext|>\n")
        made += 1
        if made >= 3:
            break
    return made


# ---------------------------------------------------------------------- main
def process(lang, budget_bytes, out_dir, cache_dir, bridge_fp=None):
    import pyarrow.parquet as pq

    out_path = os.path.join(out_dir, f"{lang}.txt")
    dedup = Dedup()
    rng = random.Random(1234)
    written = kept = seen = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for dataset, config, column, _w in SOURCES[lang]:
            if written >= budget_bytes:
                break
            log(f"  source {dataset} [{config}]")
            shards = list_shards(dataset, config)
            if not shards:
                log("    (no parquet endpoint, skipped)")
                continue
            log(f"    {len(shards)} shard(s) available")
            for i, url in enumerate(shards):
                if written >= budget_bytes:
                    break
                name = f"{lang}-{dataset.replace('/','_')}-{i}.parquet"
                path = os.path.join(cache_dir, name)
                log(f"    downloading shard {i} -> {name}")
                try:
                    size = download(url, path)
                except Exception as e:
                    log(f"    download failed: {e}")
                    continue
                log(f"    {size/1048576:.0f} MB, reading")
                try:
                    pf = pq.ParquetFile(path)
                except Exception as e:
                    log(f"    parquet error: {e}")
                    continue
                col = column if column in pf.schema.names else pf.schema.names[-1]
                for batch in pf.iter_batches(batch_size=512, columns=[col]):
                    for value in batch.column(0):
                        if written >= budget_bytes:
                            break
                        text = value.as_py()
                        if not text:
                            continue
                        seen += 1
                        if lang == "py":
                            ok, ratio, _ = python_quality(text)
                            if not ok or not dedup.accept(text):
                                continue
                            out.write(text.rstrip() + "\n\n")
                            written += len(text)
                            kept += 1
                            if bridge_fp is not None:
                                mine_bridge(text, rng, bridge_fp)
                        else:
                            text = clean_prose(text)
                            if not keep_prose(text, lang) or not dedup.accept(text):
                                continue
                            out.write(text + "\n\n")
                            written += len(text)
                            kept += 1
                        if kept % 20000 == 0:
                            log(f"      kept {kept} docs, {written/1048576:.0f} MB")
                    if written >= budget_bytes:
                        break
    log(f"  => {out_path}: {written/1048576:.1f} MB from {kept}/{seen} docs "
        f"(dedup: {dedup.dropped_exact} exact, {dedup.dropped_near} near)")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--cache-dir", default="data/raw")
    ap.add_argument("--budget-mb", type=float, default=900.0,
                    help="total cleaned output budget across languages")
    ap.add_argument("--fa", action="store_true")
    ap.add_argument("--en", action="store_true")
    ap.add_argument("--py", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--share", default="fa=0.5,en=0.2,py=0.3",
                    help="share of the budget per language")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for lang, srcs in SOURCES.items():
            print(f"{lang}:")
            for d, c, col, _ in srcs:
                print(f"  {d} [{c}] column={col}")
        return 0

    langs = [l for l in ("fa", "en", "py") if getattr(args, l) or args.all]
    if not langs:
        langs = ["fa", "en", "py"]
    shares = {}
    for item in args.share.split(","):
        k, _, v = item.partition("=")
        shares[k.strip()] = float(v)
    total_share = sum(shares.get(l, 0.0) for l in langs) or 1.0

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    bridge_path = os.path.join(args.out_dir, "bridge.txt")
    bridge_fp = open(bridge_path, "w", encoding="utf-8") if "py" in langs else None

    stats = {}
    for lang in langs:
        budget = args.budget_mb * 1048576 * shares.get(lang, 0.0) / total_share
        log(f"[{lang}] budget {budget/1048576:.0f} MB")
        stats[lang] = process(lang, budget, args.out_dir, args.cache_dir, bridge_fp)
    if bridge_fp:
        bridge_fp.close()
        stats["bridge"] = os.path.getsize(bridge_path)
        log(f"  => {bridge_path}: {stats['bridge']/1048576:.1f} MB of mined "
            f"(docstring -> code) instruction pairs")

    log("\nsummary")
    for k, v in stats.items():
        log(f"  {k:8s} {v/1048576:8.1f} MB")
    log("\nnext: ./scripts/update_data.sh   (tokenise + binary cache + stats)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
