import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

DATA_DIR = ""
TRAIN_PATH = os.path.join(DATA_DIR, "train.parquet")
VAL_PATH   = os.path.join(DATA_DIR, "validation.parquet")
TEST_PATH  = os.path.join(DATA_DIR, "test.parquet")

OUT_DIR = "./outputs_baseline_tfidf_lr_A"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Config
# -----------------------------
MAX_CHARS = 10000
TFIDF_MAX_FEATURES = 100000
NGRAM_RANGE = (2, 5)
LR_C = 1.0
LR_MAX_ITER = 1000

# -----------------------------
# Load data
# -----------------------------
print("Loading data...")
train_df = pd.read_parquet(TRAIN_PATH)
val_df = pd.read_parquet(VAL_PATH)
test_df = pd.read_parquet(TEST_PATH)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

train_codes = train_df["code"].fillna("").str[:MAX_CHARS].tolist()
val_codes = val_df["code"].fillna("").str[:MAX_CHARS].tolist()
test_codes = test_df["code"].fillna("").str[:MAX_CHARS].tolist()

train_labels = train_df["label"].values
val_labels = val_df["label"].values

# -----------------------------
# TF-IDF
# -----------------------------
print("Fitting TF-IDF...")
tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=NGRAM_RANGE,
    max_features=TFIDF_MAX_FEATURES,
    sublinear_tf=True,
    dtype=np.float32,
)
X_train = tfidf.fit_transform(train_codes)
print(f"TF-IDF matrix: {X_train.shape}")

X_val = tfidf.transform(val_codes)
X_test = tfidf.transform(test_codes)

# -----------------------------
# Logistic Regression
# -----------------------------
print("Training Logistic Regression...")
clf = LogisticRegression(
    C=LR_C,
    max_iter=LR_MAX_ITER,
    solver="lbfgs",
    n_jobs=-1,
    verbose=1,
)
clf.fit(X_train, train_labels)

# Validation
val_preds = clf.predict(X_val)
val_f1 = f1_score(val_labels, val_preds, average="macro")
print(f"Validation macro-F1: {val_f1:.4f}")

# Test predictions
print("Predicting on test set...")
test_preds = clf.predict(X_test)

# -----------------------------
# Save submission
# -----------------------------
id_col = "id" if "id" in test_df.columns else "ID"
sub = pd.DataFrame({id_col: test_df[id_col].values, "label": test_preds})
sub_path = os.path.join(OUT_DIR, "submission_baseline_tfidf_lr.csv")
sub.to_csv(sub_path, index=False)
print(f"Saved submission to: {sub_path}")
print(f"Shape: {sub.shape}")
print(sub.head())
print(f"\nPrediction distribution: {pd.Series(test_preds).value_counts().sort_index().to_dict()}")