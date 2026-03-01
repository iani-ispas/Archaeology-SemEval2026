import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import inspect
import sys, types

try:
    from accelerate import Accelerator
    if not hasattr(Accelerator, "_orig_unwrap_model"):
        Accelerator._orig_unwrap_model = Accelerator.unwrap_model
    _orig = Accelerator._orig_unwrap_model
    if "keep_torch_compile" not in inspect.signature(_orig).parameters:
        def _unwrap_model_compat(self, model, *args, **kwargs):
            return _orig(self, model)
        Accelerator.unwrap_model = _unwrap_model_compat
except Exception:
    pass

try:
    import peft  # noqa: F401
    if not hasattr(peft, "PeftModel") or not hasattr(peft, "PeftMixedModel"):
        raise ImportError("peft missing symbols")
except Exception:
    peft_stub = types.ModuleType("peft")
    class PeftModel: pass
    class PeftMixedModel: pass
    peft_stub.PeftModel = PeftModel
    peft_stub.PeftMixedModel = PeftMixedModel
    sys.modules["peft"] = peft_stub

import re, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
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

DATA_DIR = ""
TRAIN_PATH = os.path.join(DATA_DIR, "train.parquet")
VAL_PATH   = os.path.join(DATA_DIR, "validation.parquet")
TEST_PATH  = os.path.join(DATA_DIR, "test.parquet")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

OUT_ROOT = "./outputs_taskB_unixcoder_1024"
os.makedirs(OUT_ROOT, exist_ok=True)

for p in [TRAIN_PATH, VAL_PATH, TEST_PATH, SAMPLE_SUB_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}")

print("TRAIN :", TRAIN_PATH)
print("VAL   :", VAL_PATH)
print("TEST  :", TEST_PATH)
print("SAMPLE:", SAMPLE_SUB_PATH)
print("OUT_ROOT:", OUT_ROOT)


MODEL_NAME = "microsoft/unixcoder-base"
NUM_LABELS = 11

SEEDS = [42, 43, 44]

MAX_LENGTH = 1024
EPOCHS = 3.0
LR = 1.5e-5
BATCH_SIZE = 4
GRAD_ACCUM = 8
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
LABEL_SMOOTHING = 0.0
EARLYSTOP_PATIENCE = 1

USE_CLASS_WEIGHTS = True
CB_BETA = 0.9995

UNIXCODER_MODE = "<encoder-only>\n"
DELIM = "\n/*<MID_SNIP>*/\n"

PRETOKEN_MAX_CHARS = 24000
DELIM_CHAR = "\n/*<TRUNC>*/\n"

USE_OVERLAP_OVERRIDE = True
HASH_CHUNK = 50000

INFER_BATCH_SIZE = 48
INFER_NUM_WORKERS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def hash_codes_fast(codes):
    s = pd.Series(codes, dtype="object")
    return pd.util.hash_pandas_object(s, index=False).astype("uint64").to_numpy()

def build_hash_to_label(train_ds, val_ds, chunk=50000):
    h2y = {}
    for split_name, dset in [("train", train_ds), ("val", val_ds)]:
        for i in tqdm(range(0, len(dset), chunk), desc=f"Hashing {split_name}"):
            batch = dset[i:i+chunk]
            codes = [(c or "").replace("\x00", "") for c in batch["code"]]
            labels = batch["label"]
            hs = hash_codes_fast(codes)
            for h, y in zip(hs, labels):
                h2y[int(h)] = int(y)
    return h2y

def compute_test_hashes(test_ds, chunk=50000):
    out = np.zeros((len(test_ds),), dtype=np.uint64)
    for i in tqdm(range(0, len(test_ds), chunk), desc="Hashing test"):
        batch = test_ds[i:i+chunk]
        codes = [(c or "").replace("\x00", "") for c in batch["code"]]
        out[i:i+len(codes)] = hash_codes_fast(codes)
    return out

# -----------------------------
# Utils: prepare code + sandwich pack tokens
# -----------------------------
def prepare_code(code: str, max_chars: int) -> str:
    s = (code or "").replace("\x00", "")
    if len(s) <= max_chars:
        return s if s.strip() else "\n"
    half = max_chars // 2
    return s[:half] + DELIM_CHAR + s[-(max_chars - half):]

def pack_head_tail_tokens(tokenizer, text: str, max_body_tokens: int, head_frac: float = 0.60):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_body_tokens:
        return ids

    delim_ids = tokenizer.encode(DELIM, add_special_tokens=False)
    if len(delim_ids) > 16:
        delim_ids = delim_ids[:16]

    avail = max_body_tokens - len(delim_ids)
    if avail <= 16:
        return ids[:max_body_tokens]

    head_len = int(avail * head_frac)
    tail_len = avail - head_len

    head_ids = ids[:head_len]
    tail_ids = ids[-tail_len:] if tail_len > 0 else []
    return head_ids + delim_ids + tail_ids

# -----------------------------
# Load datasets
# -----------------------------
ds_tv = load_dataset("parquet", data_files={"train": TRAIN_PATH, "validation": VAL_PATH})
ds_test = load_dataset("parquet", data_files={"test": TEST_PATH})["test"]

for split in ["train", "validation"]:
    if "__index_level_0__" in ds_tv[split].column_names:
        ds_tv[split] = ds_tv[split].remove_columns(["__index_level_0__"])
if "__index_level_0__" in ds_test.column_names:
    ds_test = ds_test.remove_columns(["__index_level_0__"])

ds = DatasetDict({"train": ds_tv["train"], "validation": ds_tv["validation"], "test": ds_test})

def keep_only(split: str):
    cols = ds[split].column_names
    if split in ["train", "validation"]:
        keep = {"code", "label"}
    else:
        keep = {"code", "id", "ID"}
    rem = [c for c in cols if c not in keep]
    if rem:
        ds[split] = ds[split].remove_columns(rem)

keep_only("train")
keep_only("validation")
keep_only("test")

test_cols = ds["test"].column_names
ID_COL = "id" if "id" in test_cols else ("ID" if "ID" in test_cols else None)
if ID_COL is None:
    raise ValueError(f"Test must contain 'id' or 'ID'. Columns: {test_cols}")

print(ds)
print("Using ID column:", ID_COL)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# -----------------------------
# Collators: sandwich packing everywhere
# -----------------------------
class SandwichTrainCollator:
    def __init__(self, tok, max_length, mode_token, pretoken_max_chars):
        self.tok = tok
        self.max_length = max_length
        self.pretoken_max_chars = pretoken_max_chars

        self.cls_id = self.tok.cls_token_id or self.tok.bos_token_id
        self.sep_id = self.tok.sep_token_id or self.tok.eos_token_id
        self.mode_ids = self.tok.encode(mode_token, add_special_tokens=False)

    def __call__(self, features):
        input_ids, attn, labels = [], [], []
        for ex in features:
            code = prepare_code(ex["code"], self.pretoken_max_chars)
            head_frac = random.choice([0.50, 0.60, 0.70])
            packed = pack_head_tail_tokens(
                self.tok,
                code,
                max_body_tokens=self.max_length - 2 - len(self.mode_ids),
                head_frac=head_frac,
            )

            ids = [self.cls_id] + self.mode_ids + packed + [self.sep_id]
            input_ids.append(ids)
            attn.append([1] * len(ids))
            labels.append(int(ex["label"]))

        batch = self.tok.pad(
            {"input_ids": input_ids, "attention_mask": attn},
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

class SandwichEvalCollator:
    def __init__(self, tok, max_length, mode_token, pretoken_max_chars):
        self.tok = tok
        self.max_length = max_length
        self.pretoken_max_chars = pretoken_max_chars

        self.cls_id = self.tok.cls_token_id or self.tok.bos_token_id
        self.sep_id = self.tok.sep_token_id or self.tok.eos_token_id
        self.mode_ids = self.tok.encode(mode_token, add_special_tokens=False)

    def __call__(self, features):
        input_ids, attn, labels = [], [], []
        for ex in features:
            code = prepare_code(ex["code"], self.pretoken_max_chars)
            packed = pack_head_tail_tokens(
                self.tok,
                code,
                max_body_tokens=self.max_length - 2 - len(self.mode_ids),
                head_frac=0.60,  # deterministic for eval
            )

            ids = [self.cls_id] + self.mode_ids + packed + [self.sep_id]
            input_ids.append(ids)
            attn.append([1] * len(ids))
            labels.append(int(ex["label"]))

        batch = self.tok.pad(
            {"input_ids": input_ids, "attention_mask": attn},
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


train_collator = SandwichTrainCollator(tokenizer, MAX_LENGTH, UNIXCODER_MODE, PRETOKEN_MAX_CHARS)
eval_collator  = SandwichEvalCollator(tokenizer, MAX_LENGTH, UNIXCODER_MODE, PRETOKEN_MAX_CHARS)

# -----------------------------
# Class-balanced weights
# -----------------------------
train_labels = np.array(ds["train"]["label"], dtype=np.int64)
counts = np.bincount(train_labels, minlength=NUM_LABELS).astype(np.float64)
counts[counts == 0] = 1.0

beta = CB_BETA
effective_num = 1.0 - np.power(beta, counts)
cb_weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
cb_weights = cb_weights / cb_weights.mean()
cb_weights_t = torch.tensor(cb_weights, dtype=torch.float32)

print("Train label counts:", counts.astype(int))
print("CB weights:", np.round(cb_weights, 3))

# -----------------------------
# Trainer
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, label_smoothing=0.0, eval_collator=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self._eval_collator = eval_collator

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
            label_smoothing=self.label_smoothing,
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self._eval_collator if self._eval_collator is not None else self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

def build_training_args(**kwargs):
    sig = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    if "eval_strategy" in sig and "eval_strategy" not in kwargs and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs["evaluation_strategy"]
    if "evaluation_strategy" in sig and "evaluation_strategy" not in kwargs and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs["eval_strategy"]

    filt = {k: v for k, v in kwargs.items() if k in sig}
    return TrainingArguments(**filt)

# -----------------------------
# Overlap override: build maps once
# -----------------------------
if USE_OVERLAP_OVERRIDE:
    h2y = build_hash_to_label(ds["train"], ds["validation"], chunk=HASH_CHUNK)
    test_hashes = compute_test_hashes(ds["test"], chunk=HASH_CHUNK)
    print(f"Built hash map size={len(h2y):,}. Test hashes={len(test_hashes):,}.")
else:
    h2y, test_hashes = None, None

# -----------------------------
# Inference: single-pass sandwich input
# -----------------------------
class TestDataset(Dataset):
    def __init__(self, hf_ds, id_col):
        self.ds = hf_ds
        self.id_col = id_col

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        return {"idx": i, "ID": ex[self.id_col], "code": ex["code"]}


# build these once (near tokenizer init)
CLS_ID = tokenizer.cls_token_id or tokenizer.bos_token_id
SEP_ID = tokenizer.sep_token_id or tokenizer.eos_token_id
MODE_IDS = tokenizer.encode(UNIXCODER_MODE, add_special_tokens=False)

def make_infer_collate(head_frac: float):
    def infer_collate(batch):
        idxs  = torch.tensor([b["idx"] for b in batch], dtype=torch.long)
        ids   = torch.tensor([b["ID"] for b in batch], dtype=torch.long)

        input_ids, attn = [], []
        for b in batch:
            code = prepare_code(b["code"], PRETOKEN_MAX_CHARS)
            packed = pack_head_tail_tokens(
                tokenizer,
                code,
                max_body_tokens=MAX_LENGTH - 2 - len(MODE_IDS),
                head_frac=head_frac,
            )
            ids_i = [CLS_ID] + MODE_IDS + packed + [SEP_ID]
            input_ids.append(ids_i)
            attn.append([1] * len(ids_i))

        enc = tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attn},
            padding=True,
            return_tensors="pt",
        )
        enc["idx"] = idxs
        enc["ID"] = ids
        return enc
    return infer_collate



def predict_probs(model, test_ds, collate_fn):
    dl = DataLoader(
        test_ds,
        batch_size=INFER_BATCH_SIZE,
        shuffle=False,
        num_workers=INFER_NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    N = len(test_ds)
    sum_probs = torch.zeros((N, NUM_LABELS), dtype=torch.float32)
    ids_store = torch.full((N,), -1, dtype=torch.long)

    model.eval()
    use_amp = torch.cuda.is_available()
    from torch.cuda.amp import autocast

    with torch.no_grad():
        for batch in tqdm(dl, desc="Infer"):
            idx = batch.pop("idx")
            ids_b = batch.pop("ID")
            ids_store[idx] = ids_b

            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1).float().cpu()
            sum_probs.index_add_(0, idx, probs)

    return sum_probs, ids_store

# -----------------------------
# Train + (optional) multi-seed ensemble
# -----------------------------
all_probs = None
final_ids_store = None

for seed in SEEDS:
    print(f"\n==================== SEED {seed} ====================")
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = os.path.join(OUT_ROOT, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        weights_only=False,
    ).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    training_args = build_training_args(
        output_dir=out_dir,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=200,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        # H100: prefer bf16
        bf16=torch.cuda.is_available(),
        fp16=False,

        dataloader_num_workers=2,
        dataloader_pin_memory=True,

        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",

        report_to="none",
        remove_unused_columns=False,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=train_collator,
        eval_collator=eval_collator,
        compute_metrics=compute_metrics,
        class_weights=cb_weights_t if USE_CLASS_WEIGHTS else None,
        label_smoothing=LABEL_SMOOTHING,
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=EARLYSTOP_PATIENCE))

    trainer.train()
    val_metrics = trainer.evaluate()
    print("VAL metrics:", val_metrics)

    test_ds = TestDataset(ds["test"], ID_COL)
    tta_fracs = [0.50, 0.60, 0.70]
    probs_sum, ids_store = None, None
    for hf in tta_fracs:
        p, ids_store = predict_probs(trainer.model, test_ds, collate_fn=make_infer_collate(hf))
        probs_sum = p if probs_sum is None else (probs_sum + p)
    probs = probs_sum / float(len(tta_fracs))

    if all_probs is None:
        all_probs = probs
        final_ids_store = ids_store
    else:
        all_probs += probs

    del trainer, model
    torch.cuda.empty_cache()

# Average ensemble
avg_probs = all_probs / float(len(SEEDS))
pred_labels = torch.argmax(avg_probs, dim=1).numpy()

if USE_OVERLAP_OVERRIDE:
    overrides = 0
    for i, h in enumerate(test_hashes):
        y = h2y.get(int(h))
        if y is not None:
            pred_labels[i] = y
            overrides += 1
    print(f"Overlap overrides applied: {overrides:,}")

sub = pd.DataFrame({ID_COL: final_ids_store.numpy(), "label": pred_labels})
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
if "id" in sample_sub.columns and ID_COL != "id":
    sub = sub.rename(columns={ID_COL: "id"})

sub_path = os.path.join(OUT_ROOT, "submission_unixcodet_1024_v2.csv")
sub.to_csv(sub_path, index=False)
print("Saved submission to:", sub_path)
print(sub.head())
print("Submission shape:", sub.shape)