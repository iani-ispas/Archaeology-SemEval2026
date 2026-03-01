import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import inspect
import sys, types

try:
    from accelerate import Accelerator

    if not hasattr(Accelerator, "_orig_unwrap_model"):
        Accelerator._orig_unwrap_model = Accelerator.unwrap_model  # store original once

    _orig = Accelerator._orig_unwrap_model
    if "keep_torch_compile" not in inspect.signature(_orig).parameters:
        def _unwrap_model_compat(self, model, *args, **kwargs):
            # ignore any unknown kwargs
            return _orig(self, model)


        Accelerator.unwrap_model = _unwrap_model_compat
except Exception:
    pass

try:
    import peft

    if not hasattr(peft, "PeftModel") or not hasattr(peft, "PeftMixedModel"):
        raise ImportError("peft missing symbols")
except Exception:
    peft_stub = types.ModuleType("peft")


    class PeftModel:
        pass


    class PeftMixedModel:
        pass


    peft_stub.PeftModel = PeftModel
    peft_stub.PeftMixedModel = PeftMixedModel
    sys.modules["peft"] = peft_stub


try:
    import transformers
    from transformers.trainer import Trainer as _Trainer
except Exception:
    pass

import re, random, warnings

warnings.filterwarnings("ignore")
from collections import defaultdict

import re
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

import pyarrow.parquet as pq
from tqdm.auto import tqdm
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    EarlyStoppingCallback,
)
from tqdm.auto import tqdm


def escape_t5_special_tokens(text: str, tokenizer: AutoTokenizer) -> str:
    try:
        special = tokenizer.all_special_tokens
    except Exception:
        special = []
    for tok in special:
        if tok and tok in text:
            # Break exact match so it's not treated as a special token
            text = text.replace(tok, tok.replace("<", "< ").replace(">", " >"))
    return text

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = ""
TRAIN_PATH = os.path.join(DATA_DIR, "train.parquet")
VAL_PATH = os.path.join(DATA_DIR, "validation.parquet")

TEST_PATH_CANDIDATES = [
    os.path.join(DATA_DIR, "test.parquet"),
    os.path.join(DATA_DIR, "test_a.parquet"),
]
DIFFICULT_TEST_PATH_CANDIDATES = [
    os.path.join(DATA_DIR, "difficult_test.parquet"),
    os.path.join(DATA_DIR, "difficult_test_a.parquet"),
]

OUT_DIR = "./CodeT5-220M"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "Salesforce/codet5p-220m"
SEEN_LANGS = ["Python", "C++", "Java"]
NUM_LABELS = 2

MAX_LEN = 256
BATCH_SIZE = 16
GRAD_ACCUM = 2
LR = 3e-5
EPOCHS = 1
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
FP16 = True

TRAIN_CHUNK_CHARS = 1024
INFER_CHUNK_CHARS = 1024
INFER_OVERLAP_CHARS = 150

MAX_CHUNKS_PER_CODE = 6
FILE_BATCH_ROWS = 256
CHUNK_BATCH_SIZE = 80
PARQUET_BATCH_ROWS = 2048

FAST_6H = True
BALANCE_TRAIN_LANG_LABEL = True
MAX_OOF_PER_FOLD = 50_000
TARGET_PER_LANG_PY_FOLDS = 80_000
TARGET_PER_LANG_SMALL_FOLDS = 40_000
USE_OOF_THRESHOLD = True

FREEZE_BOTTOM_LAYERS = False
N_FREEZE = 8

# augmentation knobs (OOD-oriented)
P_STRIP_COMMENTS = 0.10
P_MASK_LITERALS = 0.15
P_DROP_HEADER = 0.10

# seeds
SEEDS = [1337]
SEED = SEEDS[0]

# -----------------------------
# NORMALIZATION / AUGMENTATION
# -----------------------------
_comment_block_re = re.compile(r"/\*.*?\*/", re.DOTALL)
_comment_line_cpp_re = re.compile(r"//.*?$", re.MULTILINE)
_comment_line_hash_full_re = re.compile(r"^\s*#.*?$", re.MULTILINE)

_string_double_re = re.compile(r'"(?:\\.|[^"\\])*"')
_string_single_re = re.compile(r"'(?:\\.|[^'\\])*'")
_number_re = re.compile(r"\b\d+(\.\d+)?\b")

_fence_line_re = re.compile(r"^\s*```.*?$", re.MULTILINE)


def basic_clean(s: str) -> str:
    """Deterministic cleanup applied to train/val/test identically."""
    s = "" if s is None else str(s)
    # normalize line endings
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # remove markdown fence lines
    s = _fence_line_re.sub("", s)
    # normalize tabs
    s = s.replace("\t", "    ")
    # trim trailing whitespace per line
    s = "\n".join([ln.rstrip() for ln in s.split("\n")])
    # compress excessive blank lines
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def strip_comments(s: str) -> str:
    # block comments (C/JS/PHP)
    s = _comment_block_re.sub(" ", s)
    # // comments
    s = _comment_line_cpp_re.sub(" ", s)
    # full-line # comments (avoid stripping inline python code too aggressively)
    s = _comment_line_hash_full_re.sub(" ", s)
    return s


def mask_literals(s: str) -> str:
    s = _string_double_re.sub('" <STR> "', s)
    s = _string_single_re.sub("' <STR> '", s)
    s = _number_re.sub(" <NUM> ", s)
    return s


def drop_header_block(s: str, max_lines: int = 40) -> str:
    lines = s.split("\n")
    cut = 0
    for i, ln in enumerate(lines[:max_lines]):
        t = ln.strip()
        if t == "":
            cut = i + 1
            continue
        if t.startswith(("#", "//", "/*", "*", '"""', "'''")):
            cut = i + 1
            continue
        if t.startswith(("import ", "from ", "package ", "using ", "require(", "const ", "var ", "let ")):
            cut = i + 1
            continue
        break
    return "\n".join(lines[cut:]) if cut > 0 else s


def sample_train_chunk(s: str, chunk_chars: int) -> str:
    s = basic_clean(s)
    if len(s) <= chunk_chars:
        return s
    r = random.random()
    if r < 0.50:
        start = 0
    elif r < 0.80:
        start = random.randint(0, len(s) - chunk_chars)
    else:
        start = max(0, len(s) - chunk_chars)
    return s[start: start + chunk_chars]


# -----------------------------
# HELPERS: sampling to control runtime
# -----------------------------
def cap_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df
    return df.sample(n, random_state=seed).reset_index(drop=True)


def make_balanced_lang_label(df: pd.DataFrame, langs: List[str], n_per_lang: int, seed: int) -> pd.DataFrame:
    parts = []
    n0 = n_per_lang // 2
    n1 = n_per_lang - n0

    for lang in langs:
        d = df[df["language"] == lang]
        for lab, n in [(0, n0), (1, n1)]:
            dl = d[d["label"] == lab]
            if len(dl) == 0:
                continue
            replace = len(dl) < n
            rs = seed + (abs(hash(lang)) % 10000) + 100 * lab
            parts.append(dl.sample(n, replace=replace, random_state=rs))

    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


# -----------------------------
# Language inference (cheap heuristics for per-slice thresholding)
# -----------------------------
_lang_rules = [
    ("PHP", lambda s: "<?php" in s),
    ("Go", lambda s: ("package main" in s) or ("\nfunc " in s) or s.strip().startswith("package ")),
    ("C#", lambda s: ("using System" in s) or ("namespace " in s and "{" in s)),
    ("JavaScript",
     lambda s: ("console.log" in s) or ("=>" in s) or ("\nfunction " in s) or ("\nconst " in s) or ("\nlet " in s)),
    ("C++", lambda s: ("std::" in s) or ("#include <" in s and ("cout" in s or "cin" in s))),
    ("C", lambda s: ("#include <" in s and "printf(" in s)),
    ("Java", lambda s: ("public static void main" in s) or ("System.out." in s)),
    ("Python", lambda s: ("\ndef " in s) or ("\nclass " in s and ":" in s) or ("\nimport " in s) or (
                "\nfrom " in s and " import " in s)),
]


def infer_lang_fast(code: str) -> str:
    s = basic_clean(code)
    s = s[:4000]
    for name, fn in _lang_rules:
        try:
            if fn(s):
                return name
        except Exception:
            pass
    return "UNK"


# -----------------------------
# DATASET (training)
# -----------------------------
class CodeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, train_mode: bool, max_len: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.train_mode = train_mode
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        code = "" if row["code"] is None else str(row["code"])
        label = int(row["label"])

        if self.train_mode:
            code = sample_train_chunk(code, TRAIN_CHUNK_CHARS)
            # OOD-oriented augmentations (after deterministic clean + crop)
            if random.random() < P_DROP_HEADER:
                code = drop_header_block(code)
            if random.random() < P_STRIP_COMMENTS:
                code = strip_comments(code)
            if random.random() < P_MASK_LITERALS:
                code = mask_literals(code)
            # final tidy (don’t destroy indentation)
            code = basic_clean(code)
        else:
            code = basic_clean(code)

        # Escape literal special tokens in code (e.g. </s> in HTML/XML)
        code = escape_t5_special_tokens(code, self.tokenizer)

        enc = self.tokenizer(
            code,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# -----------------------------
# THRESHOLD SEARCH
# -----------------------------
def best_threshold_macro_f1(y_true: np.ndarray, p_ai: np.ndarray) -> Tuple[float, float]:
    thresholds = np.unique(np.quantile(p_ai, np.linspace(0.01, 0.99, 99)))
    best_t, best_f = 0.5, -1.0
    for t in thresholds:
        y_pred = (p_ai >= t).astype(int)
        f = f1_score(y_true, y_pred, average="macro")
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


# -----------------------------
# INFERENCE: deterministic chunks + trimmed-mean aggregation
# -----------------------------
def make_chunks_deterministic(
        s: str,
        chunk_chars: int,
        overlap_chars: int,
        max_chunks: int,
) -> List[str]:
    s = basic_clean(s)
    n = len(s)
    if n <= chunk_chars:
        return [s]

    # always head + tail
    starts = [0, max(0, n - chunk_chars)]
    step = max(1, chunk_chars - overlap_chars)
    mids = list(range(step, max(step, n - chunk_chars), step))

    slots = max_chunks - len(starts)
    if slots > 0 and len(mids) > 0:
        if len(mids) <= slots:
            starts += mids
        else:
            idx = np.linspace(0, len(mids) - 1, slots).round().astype(int)
            starts += [mids[i] for i in idx]

    starts = sorted(set(starts))[:max_chunks]
    return [s[st: st + chunk_chars] for st in starts]


def trimmed_mean_tensor(x: torch.Tensor) -> float:
    if x.numel() <= 2:
        return float(x.mean().item())
    vals, _ = torch.sort(x)
    vals = vals[1:-1]
    return float(vals.mean().item())


def topk_mean_tensor(x: torch.Tensor, k: int = 2) -> float:
    if x.numel() == 0:
        return 0.0
    k = min(k, x.numel())
    vals, _ = torch.topk(x, k)
    return float(vals.mean().item())


@torch.inference_mode()
def predict_proba_ai_chunked_trimmedmean_fast(
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        codes: List[str],
        device: torch.device,
        max_len: int,
        chunk_chars: int,
        overlap_chars: int,
        max_chunks_per_code: int,
        file_batch_rows: int,
        chunk_batch_size: int,
) -> np.ndarray:
    model.eval()
    out = np.zeros(len(codes), dtype=np.float32)
    use_amp = (device.type == "cuda")

    for base in range(0, len(codes), file_batch_rows):
        block = codes[base: base + file_batch_rows]

        flat_chunks: List[str] = []
        offsets = [0]
        for c in block:
            parts = make_chunks_deterministic(c, chunk_chars, overlap_chars, max_chunks_per_code)
            flat_chunks.extend(parts)
            offsets.append(len(flat_chunks))

        p1_list = []
        for i in range(0, len(flat_chunks), chunk_batch_size):
            batch = flat_chunks[i: i + chunk_batch_size]
            batch = [escape_t5_special_tokens(c, tokenizer) for c in batch]
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(**enc).logits
            else:
                logits = model(**enc).logits

            probs = torch.softmax(logits, dim=-1)
            p1_list.append(probs[:, 1].float().cpu())

        p1 = torch.cat(p1_list, dim=0)

        for j in range(len(block)):
            a, b = offsets[j], offsets[j + 1]
            tm = trimmed_mean_tensor(p1[a:b])
            t2 = topk_mean_tensor(p1[a:b], k=2)
            out[base + j] = 0.5 * tm + 0.5 * t2

    return out


# -----------------------------
# TrainingArguments compat (eval_strategy vs evaluation_strategy)
# -----------------------------
def make_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in sig and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    if "evaluation_strategy" in sig and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(**kwargs)


def freeze_bottom_layers(model: AutoModelForSequenceClassification, n_freeze: int = 8) -> None:
    if hasattr(model, "encoder"):
        enc = model.encoder
        if hasattr(enc, "embed_tokens"):
            for p in enc.embed_tokens.parameters():
                p.requires_grad = False
        if hasattr(enc, "block"):
            for i, block in enumerate(enc.block):
                if i < n_freeze:
                    for p in block.parameters():
                        p.requires_grad = False
        return
    if hasattr(model, "roberta"):
        for p in model.roberta.embeddings.parameters():
            p.requires_grad = False
        if hasattr(model.roberta, "encoder") and hasattr(model.roberta.encoder, "layer"):
            for i, layer in enumerate(model.roberta.encoder.layer):
                if i < n_freeze:
                    for p in layer.parameters():
                        p.requires_grad = False


# -----------------------------
# TRAIN ONE FOLD
# -----------------------------
def train_fold(fold_name: str, train_df: pd.DataFrame, seed: int, out_dir: str) -> str:
    set_seed(seed)
    fold_dir = os.path.join(out_dir, f"{fold_name}_seed{seed}")
    os.makedirs(fold_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS, weights_only=False)

    if FREEZE_BOTTOM_LAYERS:
        freeze_bottom_layers(model, n_freeze=N_FREEZE)

    train_ds = CodeDataset(train_df, tokenizer, train_mode=True, max_len=MAX_LEN)

    args = make_training_args(
        output_dir=fold_dir,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        label_smoothing_factor=0.02,
        logging_steps=200,
        eval_strategy="no",
        save_strategy="no",
        fp16=FP16,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )
    trainer.train()

    best_path = os.path.join(fold_dir, "best_model")
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)
    return best_path


# -----------------------------
# STREAMING PARQUET READER
# -----------------------------
def iter_test_batches_parquet(test_path: str, batch_rows: int):
    pf = pq.ParquetFile(test_path)
    cols = pf.schema.names
    id_col = "ID" if "ID" in cols else ("id" if "id" in cols else None)
    if id_col is None or "code" not in cols:
        raise ValueError(f"Test parquet must have ID/id and code. Found: {cols}")
    for rb in pf.iter_batches(batch_size=batch_rows, columns=[id_col, "code"]):
        yield id_col, rb.to_pandas()


def pick_existing_path(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# -----------------------------
# MAIN
# -----------------------------
def main():
    test_path = pick_existing_path(TEST_PATH_CANDIDATES)
    difficult_test_path = pick_existing_path(DIFFICULT_TEST_PATH_CANDIDATES)

    for p in [TRAIN_PATH, VAL_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")
    if test_path is None:
        raise FileNotFoundError(f"Missing test parquet. Tried: {TEST_PATH_CANDIDATES}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(
        "MAX_LEN:", MAX_LEN,
        "| TRAIN_CHUNK_CHARS:", TRAIN_CHUNK_CHARS,
        "| INFER_CHUNK_CHARS:", INFER_CHUNK_CHARS,
        "| MAX_CHUNKS_PER_CODE:", MAX_CHUNKS_PER_CODE,
        "| PARQUET_BATCH_ROWS:", PARQUET_BATCH_ROWS,
    )

    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)

    train_df = train_df[train_df["code"].notna()].copy()
    val_df = val_df[val_df["code"].notna()].copy()
    train_df = train_df[train_df["code"].astype(str).str.len() > 0].copy()
    val_df = val_df[val_df["code"].astype(str).str.len() > 0].copy()

    train_df = train_df[train_df["language"].isin(SEEN_LANGS)].reset_index(drop=True)
    val_df = val_df[val_df["language"].isin(SEEN_LANGS)].reset_index(drop=True)
    print("Rows train_df:", len(train_df), "| Rows val_df:", len(val_df))

    model_paths: List[str] = []
    oof_true: List[np.ndarray] = []
    oof_pred: List[np.ndarray] = []

    for seed in SEEDS:
        for holdout_lang in SEEN_LANGS:
            fold_name = f"holdout_{holdout_lang.replace('+', 'p')}"
            tr = train_df[train_df["language"] != holdout_lang].reset_index(drop=True)
            va = val_df[val_df["language"] == holdout_lang].reset_index(drop=True)

            if FAST_6H:
                langs_fold = sorted(tr["language"].unique().tolist())
                fold_seed = seed + (abs(hash(holdout_lang)) % 10_000)
                target = TARGET_PER_LANG_PY_FOLDS if "Python" in langs_fold else TARGET_PER_LANG_SMALL_FOLDS
                tr = make_balanced_lang_label(tr, langs_fold, target, fold_seed)

            print(f"\n=== Training {fold_name} seed={seed} | train_rows={len(tr)} | holdout_rows={len(va)} ===")
            best_dir = train_fold(fold_name, tr, seed, OUT_DIR)
            model_paths.append(best_dir)

            if USE_OOF_THRESHOLD:
                va_oof = cap_df(va, MAX_OOF_PER_FOLD, seed) if FAST_6H else va
                print(f"--- OOF for threshold on {len(va_oof)} examples (capped) ---")

                tok = AutoTokenizer.from_pretrained(best_dir, use_fast=True)
                mdl = AutoModelForSequenceClassification.from_pretrained(best_dir, weights_only=False).to(device)

                p_ai = predict_proba_ai_chunked_trimmedmean_fast(
                    model=mdl,
                    tokenizer=tok,
                    codes=va_oof["code"].astype(str).tolist(),
                    device=device,
                    max_len=MAX_LEN,
                    chunk_chars=INFER_CHUNK_CHARS,
                    overlap_chars=INFER_OVERLAP_CHARS,
                    max_chunks_per_code=MAX_CHUNKS_PER_CODE,
                    file_batch_rows=FILE_BATCH_ROWS,
                    chunk_batch_size=CHUNK_BATCH_SIZE,
                )
                oof_true.append(va_oof["label"].to_numpy())
                oof_pred.append(p_ai)

    if USE_OOF_THRESHOLD:
        y = np.concatenate(oof_true)
        p = np.concatenate(oof_pred)
        thr_oof, f_best_oof = best_threshold_macro_f1(y, p)
        print(f"\nOOF best threshold={thr_oof:.4f}, macroF1={f_best_oof:.4f}")
    else:
        thr_oof = 0.5
        print("\nUsing fixed threshold=0.5 (OOF disabled)")

    models = []
    for mp in model_paths:
        tok = AutoTokenizer.from_pretrained(mp, use_fast=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(mp, weights_only=False).to(device)
        mdl.eval()
        models.append((mdl, tok))
    print(f"\nLoaded {len(models)} models for inference.")

    thr_final = thr_oof
    thr_by_lang = {"__GLOBAL__": float(thr_final)}
    if difficult_test_path is not None and os.path.exists(difficult_test_path):
        try:
            ts = pd.read_parquet(difficult_test_path)
            ts = ts[ts["code"].notna()].copy()
            ts = ts[ts["code"].astype(str).str.len() > 0].copy()
            y_ts = ts["label"].astype(int).to_numpy()
            codes_ts = ts["code"].astype(str).tolist()

            print(f"\n=== Threshold calibration on difficult_test: {len(ts)} rows from {difficult_test_path} ===")
            sum_p = np.zeros(len(codes_ts), dtype=np.float32)
            for mdl, tok in models:
                p_ai = predict_proba_ai_chunked_trimmedmean_fast(
                    model=mdl,
                    tokenizer=tok,
                    codes=codes_ts,
                    device=device,
                    max_len=MAX_LEN,
                    chunk_chars=INFER_CHUNK_CHARS,
                    overlap_chars=INFER_OVERLAP_CHARS,
                    max_chunks_per_code=MAX_CHUNKS_PER_CODE,
                    file_batch_rows=FILE_BATCH_ROWS,
                    chunk_batch_size=CHUNK_BATCH_SIZE,
                )
                sum_p += p_ai
            avg_p = sum_p / len(models)

            thr_ts, f_ts = best_threshold_macro_f1(y_ts, avg_p)
            print(f"difficult_test best threshold={thr_ts:.4f}, macroF1={f_ts:.4f}")

            langs_ts = [infer_lang_fast(c) for c in codes_ts]
            thr_by_lang = {}

            for lang in sorted(set(langs_ts)):
                mask = np.array([x == lang for x in langs_ts])
                if mask.sum() >= 25:  # avoid noisy thresholds on tiny groups
                    t_lang, _ = best_threshold_macro_f1(y_ts[mask], avg_p[mask])
                    thr_by_lang[lang] = float(t_lang)

            thr_by_lang["__GLOBAL__"] = float(thr_ts)
            print("Per-lang thresholds:", thr_by_lang)

            thr_final = 0.5 * thr_oof + 0.5 * thr_ts
            print(f"Final blended threshold={thr_final:.4f} (0.5*OOF + 0.5*difficult_test)")
        except Exception as e:
            print("WARNING: failed difficult_test calibration:", repr(e))
            thr_final = thr_oof
    else:
        print("\nNo difficult_test parquet found; using OOF threshold only.")
        thr_final = thr_oof

    sub_path = os.path.join(OUT_DIR, "submission_codet5p_256_noval.csv")
    if os.path.exists(sub_path):
        os.remove(sub_path)

    first_write = True
    print(f"\n=== Streaming inference from: {test_path} ===")
    for id_col, dfb in tqdm(iter_test_batches_parquet(test_path, PARQUET_BATCH_ROWS), desc="Test batches"):
        ids = dfb[id_col].to_numpy()
        codes = dfb["code"].astype(str).tolist()

        sum_p = np.zeros(len(codes), dtype=np.float32)
        for mdl, tok in models:
            p_ai = predict_proba_ai_chunked_trimmedmean_fast(
                model=mdl,
                tokenizer=tok,
                codes=codes,
                device=device,
                max_len=MAX_LEN,
                chunk_chars=INFER_CHUNK_CHARS,
                overlap_chars=INFER_OVERLAP_CHARS,
                max_chunks_per_code=MAX_CHUNKS_PER_CODE,
                file_batch_rows=FILE_BATCH_ROWS,
                chunk_batch_size=CHUNK_BATCH_SIZE,
            )
            sum_p += p_ai

        avg_p = sum_p / len(models)
        langs_blk = [infer_lang_fast(c) for c in codes]
        thr_blk = np.array([thr_by_lang.get(l, thr_by_lang.get("__GLOBAL__", thr_final)) for l in langs_blk],
                           dtype=np.float32)
        preds = (avg_p >= thr_blk).astype(np.int64)

        out_df = pd.DataFrame({"ID": ids.astype(np.int64), "label": preds.astype(np.int64)})
        out_df.to_csv(sub_path, index=False, mode="a", header=first_write)
        first_write = False

    print("\nSaved:", sub_path)
    try:
        print(pd.read_csv(sub_path).head(10))
    except Exception:
        pass


if __name__ == "__main__":
    main()