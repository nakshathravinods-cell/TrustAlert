"""
sepsisguard.py - engine for the SepsisGuard early-warning project.

Used by BOTH the Colab notebook and the Streamlit app, so features and alert
rules are identical at training time and at prediction time.

Research-integrity rules baked in:
  * every feature at hour t uses only hours <= t of the same patient (no future peeking)
  * target = SepsisLabel HORIZON hours later; rows already positive are dropped
  * splits are by PATIENT, never by row
  * alert thresholds are fitted on VALIDATION data, never on test data
  * nothing here invents data
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Column names of the PhysioNet 2019 data (Kaggle "Dataset.csv" uses the same)
# ---------------------------------------------------------------------------
ID = "Patient_ID"
TIME = "ICULOS"          # hours since ICU admission (we fall back to "Hour")
LABEL = "SepsisLabel"
HORIZON = 6              # predict the label this many hours ahead
ROLL_WINDOW = 6          # trailing window (hours) for rolling features

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]
ROLL_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]

# ---- data-reliability settings (design choices, not clinical standards) ----
KEY_VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp"]
PLAUSIBLE = {            # broad physiological limits; values outside are suspicious
    "HR": (20, 250), "O2Sat": (50, 100), "Temp": (30, 43),
    "SBP": (40, 300), "MAP": (20, 200), "Resp": (4, 60),
}
FRESH_HOURS = 6          # a vital measured >= 6 h ago counts as "stale"
COOLDOWN_HOURS = 6       # after an alert, stay quiet this long (unless it escalates)
WARMUP_HOURS = 6         # no alerts in the first 6 ICU hours: too little history to judge


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def standardise(df: pd.DataFrame, need_id: bool = True) -> pd.DataFrame:
    """Make column names match what the rest of the code expects."""
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()  # drop saved row-index column
    lower = {c.lower(): c for c in df.columns}
    ren = {}
    for target, alts in ((ID, ("patient_id", "patientid", "pid", "patient")),
                         (LABEL, ("sepsislabel",)), (TIME, ("iculos",))):
        if target not in df.columns:
            for a in alts:
                if a in lower:
                    ren[lower[a]] = target
                    break
    df = df.rename(columns=ren)
    if ID not in df.columns:
        if need_id:
            raise ValueError(
                f"No patient-identifier column found. Columns present: {list(df.columns)}. "
                f"A patient-level split needs one: rename it to '{ID}'.")
        df[ID] = "patient"
    if TIME not in df.columns:
        if "Hour" in df.columns:
            df[TIME] = df["Hour"]
        else:
            df[TIME] = df.groupby(ID).cumcount() + 1   # assume hourly rows in order
    return df


def load_data(path: str, max_patients: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Load ONE csv, or a FOLDER of per-patient .psv/.csv files (original PhysioNet layout)."""
    rng = np.random.default_rng(seed)
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.psv"), recursive=True))
        if not files:
            files = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
        if not files:
            raise FileNotFoundError(f"No .psv or .csv files found under {path}")
        if max_patients and len(files) > max_patients:
            files = sorted(rng.choice(files, max_patients, replace=False).tolist())
        frames = []
        for f in files:
            d = pd.read_csv(f, sep="|" if f.endswith(".psv") else ",")
            if ID not in d.columns:
                d[ID] = os.path.basename(os.path.dirname(f)) + "_" + os.path.splitext(os.path.basename(f))[0]
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path)
    df = standardise(df)
    if max_patients and not os.path.isdir(path):
        ids = df[ID].drop_duplicates().to_numpy()
        if len(ids) > max_patients:
            keep = rng.choice(ids, max_patients, replace=False)
            df = df[df[ID].isin(keep)]
    if LABEL not in df.columns:
        raise ValueError(f"Target column '{LABEL}' not found. Columns: {list(df.columns)}")
    return df.sort_values([ID, TIME]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Features (past-only) and data reliability
# ---------------------------------------------------------------------------
def reliability_table(filled: pd.DataFrame, since: dict) -> pd.DataFrame:
    """Score how much we can trust the data behind each hour's prediction (0..1).
       coverage     = share of key vitals that have any known value
       freshness    = how recently each key vital was really measured (1 = this hour, 0 = >= 6 h ago / never)
       plausibility = share of present key values inside broad physiological limits
       reliability  = 0.4*coverage + 0.4*freshness + 0.2*plausibility   (weights are a design choice)"""
    keys = [k for k in KEY_VITALS if k in filled.columns]
    if not keys:
        raise ValueError("None of the key vitals (HR, O2Sat, Temp, SBP, MAP, Resp) are in the data.")
    n = len(keys)
    present = sum(filled[k].notna().astype(float) for k in keys)
    coverage = present / n
    freshness = sum((1 - since[k] / FRESH_HOURS).clip(0, 1).fillna(0.0) for k in keys) / n
    bad = sum((((filled[k] < lo) | (filled[k] > hi)) & filled[k].notna()).astype(float)
              for k, (lo, hi) in PLAUSIBLE.items() if k in keys)
    plaus = 1 - bad / present.clip(lower=1)
    rel = 0.4 * coverage + 0.4 * freshness + 0.2 * plaus
    return pd.DataFrame({"coverage": coverage, "freshness": freshness,
                         "plausibility": plaus, "reliability": rel})


def build_features(df: pd.DataFrame):
    """Return (features, sorted_df, reliability). Row t uses only rows <= t of the same patient."""
    df = df.sort_values([ID, TIME]).reset_index(drop=True)
    pid = df[ID]
    cols = {TIME: df[TIME]}
    for c in STATIC:
        if c in df.columns:
            cols[c] = df[c]

    dyn = [c for c in VITALS + LABS if c in df.columns]
    filled = df[dyn].groupby(pid, sort=False).ffill()          # carry last known value forward
    for c in dyn:
        cols[c] = filled[c]
        cols[f"{c}_measured"] = df[c].notna().astype(np.int8)   # was it actually measured this hour?

    since = {}
    for c in [k for k in KEY_VITALS if k in dyn]:               # hours since last REAL measurement
        last = df[TIME].where(df[c].notna()).groupby(pid, sort=False).ffill()
        since[c] = df[TIME] - last
        cols[f"{c}_hrs_since"] = since[c]

    for c in [c for c in ROLL_COLS if c in dyn]:
        gb = filled[c].groupby(pid, sort=False)
        roll = gb.rolling(ROLL_WINDOW, min_periods=1)
        cols[f"{c}_mean{ROLL_WINDOW}"] = roll.mean().to_numpy()
        cols[f"{c}_min{ROLL_WINDOW}"] = roll.min().to_numpy()
        cols[f"{c}_max{ROLL_WINDOW}"] = roll.max().to_numpy()
        cols[f"{c}_diff1"] = gb.diff()                          # change since previous hour

    if "HR" in dyn and "SBP" in dyn:
        cols["ShockIndex"] = (filled["HR"] / filled["SBP"]).replace([np.inf, -np.inf], np.nan)

    feats = pd.DataFrame(cols, index=df.index)
    rel = reliability_table(filled, since)
    return feats, df, rel


def build_dataset(df: pd.DataFrame, horizon: int = HORIZON):
    """Early-warning table: target(t) = SepsisLabel at hour t+horizon; rows already positive at t are removed.
    Returns X (features), y (0/1 array), meta (Patient_ID, ICULOS + reliability columns)."""
    feats, d, rel = build_features(df)
    future = d.groupby(ID, sort=False)[LABEL].shift(-horizon)
    keep = (d[LABEL] == 0) & future.notna()
    X = feats.loc[keep].reset_index(drop=True)
    y = future[keep].astype(int).to_numpy()
    meta = pd.concat([d.loc[keep, [ID, TIME]], rel.loc[keep]], axis=1).reset_index(drop=True)
    return X, y, meta


def usable_features(X_train: pd.DataFrame) -> list[str]:
    """Drop columns that are completely empty in the training data."""
    return [c for c in X_train.columns if X_train[c].notna().any()]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_by_patient(meta: pd.DataFrame, y: np.ndarray, seed: int = 42,
                     val_size: float = 0.15, test_size: float = 0.15):
    """70/15/15 split where each PATIENT lands in exactly one part (stratified on 'ever positive')."""
    pat = pd.DataFrame({ID: meta[ID].to_numpy(), "y": y}).groupby(ID)["y"].max()
    ids, strat = pat.index.to_numpy(), pat.to_numpy()
    tr_ids, te_ids = train_test_split(ids, test_size=test_size, stratify=strat, random_state=seed)
    tr_ids, va_ids = train_test_split(
        tr_ids, test_size=val_size / (1 - test_size),
        stratify=pat.loc[tr_ids].to_numpy(), random_state=seed)
    assert not (set(tr_ids) & set(va_ids) or set(tr_ids) & set(te_ids) or set(va_ids) & set(te_ids))
    return (meta[ID].isin(tr_ids).to_numpy(), meta[ID].isin(va_ids).to_numpy(),
            meta[ID].isin(te_ids).to_numpy())


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def make_models(seed: int = 42) -> dict:
    lr = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    rf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=20,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed)),
    ])
    hgb = Pipeline([
        ("clf", HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
            class_weight="balanced", early_stopping=False, random_state=seed)),
    ])
    return {"Logistic Regression": lr, "Random Forest": rf, "HistGradientBoosting": hgb}


def fit_model(name, model, X_train, y_train, rf_max_rows=400_000, seed=42):
    """Random Forest trains on a random row subsample if the data is huge (speed / memory)."""
    X_fit, y_fit = X_train, np.asarray(y_train)
    if name == "Random Forest" and rf_max_rows and len(X_fit) > rf_max_rows:
        idx = np.random.default_rng(seed).choice(len(X_fit), rf_max_rows, replace=False)
        X_fit, y_fit = X_fit.iloc[idx], y_fit[idx]
    model.fit(X_fit, y_fit)
    return model


# ---------------------------------------------------------------------------
# Row-level evaluation
# ---------------------------------------------------------------------------
def pick_threshold(y, p, min_recall: float | None = None) -> float:
    """Alert threshold chosen on VALIDATION data. Default: maximise F1.
    With min_recall: the highest threshold that still catches that share of positives."""
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    if min_recall is not None:
        ok = np.where(rec >= min_recall)[0]
        return float(thr[ok[-1]]) if len(ok) else float(thr[0])
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[int(np.argmax(f1))])


def evaluate(y, p, thr: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else 0.0  # noqa: E731
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    try:
        roc = float(roc_auc_score(y, p))
    except ValueError:
        roc = float("nan")
    return {
        "threshold": float(thr), "roc_auc": roc,
        "pr_auc": float(average_precision_score(y, p)),
        "accuracy": div(tp + tn, tp + tn + fp + fn),
        "precision": precision, "recall_sensitivity": recall,
        "specificity": div(tn, tn + fp),
        "f1": div(2 * precision * recall, precision + recall),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
        "prevalence": float(y.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ---------------------------------------------------------------------------
# SepsisGuard alert layer: tier -> trend -> reliability -> priority
# ---------------------------------------------------------------------------
def fit_alert_config(meta_val: pd.DataFrame, score_val, y_val, rel_pct: float = 20,
                     min_recall_moderate: float = 0.90) -> dict:
    """Everything is learned from VALIDATION data only (never from test)."""
    score_val = np.asarray(score_val)
    thr_high = pick_threshold(y_val, score_val)                                  # best F1
    thr_mod = min(pick_threshold(y_val, score_val, min_recall=min_recall_moderate), thr_high)
    fr = pd.DataFrame({ID: meta_val[ID].to_numpy(), "score": score_val})
    d2 = fr.groupby(ID)["score"].diff(2)                                         # 2-hour change in score
    pos = d2[d2 > 0].dropna()
    return {
        "thr_high": float(thr_high), "thr_mod": float(thr_mod),
        "trend_rise": float(pos.quantile(0.75)), "trend_rapid": float(pos.quantile(0.95)),
        "rel_cut": float(np.percentile(meta_val["reliability"], rel_pct)),       # bottom rel_pct% = "unreliable"
        "cooldown": COOLDOWN_HOURS,
    }


def add_alert_layer(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """frame needs columns: Patient_ID, ICULOS, score, reliability (rows sorted by patient and time)."""
    f = frame.copy()
    d2 = f.groupby(ID)["score"].diff(2)
    f["trend"] = np.select(
        [d2.isna(), d2 >= cfg["trend_rapid"], d2 >= cfg["trend_rise"], d2 <= -cfg["trend_rise"]],
        ["INITIAL", "RAPIDLY INCREASING", "INCREASING", "DECREASING"], default="STABLE")
    f["tier"] = np.select([f["score"] >= cfg["thr_high"], f["score"] >= cfg["thr_mod"]],
                          ["HIGH", "MODERATE"], default="LOW")
    f["reliable"] = f["reliability"] >= cfg["rel_cut"]
    f["warmup"] = (f[TIME] - f.groupby(ID)[TIME].transform("min")) < WARMUP_HOURS   # first hours of a stay
    rising = f["trend"].isin(["RAPIDLY INCREASING", "INCREASING"])
    f["priority"] = np.select(
        [(f["tier"] == "HIGH") & f["reliable"],        # 3: high risk, trustworthy data
         (f["tier"] == "HIGH") & ~f["reliable"],       # 2: high risk, but check the data
         (f["tier"] == "MODERATE") & rising],          # 1: moderate and getting worse
        [3, 2, 1], default=0)
    f.loc[f["warmup"], "priority"] = 0                  # warm-up: never alert on almost no history
    f["alert_type"] = f["priority"].map({3: "HIGH-PRIORITY ALERT", 2: "HIGH RISK - VERIFY DATA",
                                         1: "RISING-RISK REVIEW", 0: "NO ALERT"})
    return f


def apply_policies(f: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Three ways of turning scores into alerts, so they can be compared fairly.
       policy_A: alert EVERY hour the score is HIGH (naive)
       policy_B: SepsisGuard - alert at the start of a high-risk episode, stay quiet for the cooldown,
                 but re-alert at once if a 'verify data' prompt escalates to a high-priority alert
       policy_C: persistence - HIGH in at least 2 of the last 3 hours, then the same cooldown"""
    f = f.sort_values([ID, TIME]).reset_index(drop=True)
    warm = f["warmup"].to_numpy() if "warmup" in f.columns else np.zeros(len(f), bool)
    high = ((f["tier"] == "HIGH").to_numpy()) & ~warm      # no alerts during warm-up
    prio = f["priority"].to_numpy()
    t = f[TIME].to_numpy(dtype=float)
    pid = f[ID].to_numpy()
    n = len(f)
    A, B, C = high.copy(), np.zeros(n, bool), np.zeros(n, bool)
    cd = cfg["cooldown"]
    bounds = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])
    for s, e in zip(bounds[:-1], bounds[1:]):
        last_b, last_pb, last_c = -1e9, 0, -1e9
        for i in range(s, e):
            if not high[i]:
                continue
            if t[i] - last_b >= cd or (prio[i] == 3 and last_pb == 2):
                B[i], last_b, last_pb = True, t[i], prio[i]
            if high[max(s, i - 2): i + 1].sum() >= 2 and t[i] - last_c >= cd:
                C[i], last_c = True, t[i]
    f["policy_A"], f["policy_B"], f["policy_C"] = A, B, C
    return f


def evaluate_policies(f: pd.DataFrame, df_sorted: pd.DataFrame,
                      policies=("policy_A", "policy_B", "policy_C")) -> pd.DataFrame:
    """Patient-level view. For each policy:
       caught_%        : sepsis patients who got >= 1 alert BEFORE their label turned 1
       median_lead_h   : hours between the first alert and the label turning 1
       alerts_per_100_patient_days : alert burden on staff
       healthy_alerted_% : patients who never got the label but received >= 1 alert (false alarms)"""
    onset = df_sorted[df_sorted[LABEL] == 1].groupby(ID)[TIME].min()
    pats = f[ID].unique()
    sep = [p for p in pats if p in onset.index]
    non = [p for p in pats if p not in onset.index]
    patient_days = len(f) / 24.0
    out = {}
    for pol in policies:
        fired = f[f[pol]]
        first = fired.groupby(ID)[TIME].min()
        lead = onset.reindex(sep) - first.reindex(sep)
        out[pol] = {
            "sepsis_patients": len(sep),
            "caught_%": float(lead.notna().mean() * 100) if sep else float("nan"),
            "median_lead_h": float(lead.median()) if lead.notna().any() else float("nan"),
            "alerts_per_100_patient_days": float(len(fired) / patient_days * 100),
            "healthy_alerted_%": float(first.reindex(non).notna().mean() * 100) if non else float("nan"),
        }
    return pd.DataFrame(out).T


# ---------------------------------------------------------------------------
# Explanation (model-agnostic "occlusion") and prediction helpers
# ---------------------------------------------------------------------------
def explain_row(model, x_row: pd.Series, medians: dict, features: list[str], top_k: int = 5) -> pd.DataFrame:
    """How much does the score drop if each feature is replaced by its training median?
    Positive = that feature pushed the score UP. Describes the MODEL, not causes of sepsis."""
    row = x_row.reindex(features).astype(float)
    base_df = pd.DataFrame([row.to_numpy()], columns=features)
    base = float(model.predict_proba(base_df)[0, 1])
    occ = pd.concat([base_df] * len(features), ignore_index=True)
    for i, f in enumerate(features):
        occ.iloc[i, i] = medians.get(f, 0.0)
    delta = base - model.predict_proba(occ)[:, 1]
    out = pd.DataFrame({"feature": features, "value": row.to_numpy(), "score_change": delta})
    out = out[out["value"].notna()]
    return out.sort_values("score_change", ascending=False).head(top_k).reset_index(drop=True)


def predict_scores(bundle: dict, X: pd.DataFrame):
    Xf = X.reindex(columns=bundle["features"])
    raw = bundle["model"].predict_proba(Xf)[:, 1]
    cal = bundle["calibrator"].predict(raw) if bundle.get("calibrator") is not None else raw
    return raw, cal


def score_history(bundle: dict, raw_df: pd.DataFrame):
    """Score every hour of one or more patients (hour t uses only data up to t) and run the alert layer.
    Returns (alert_frame, features, sorted_raw_df)."""
    raw = standardise(raw_df, need_id=False)
    feats, d, rel = build_features(raw)
    p_raw, p_cal = predict_scores(bundle, feats)
    f = pd.DataFrame({ID: d[ID].to_numpy(), TIME: d[TIME].to_numpy(), "score": p_raw,
                      "calibrated": p_cal, "reliability": rel["reliability"].to_numpy(),
                      "coverage": rel["coverage"].to_numpy(), "freshness": rel["freshness"].to_numpy(),
                      "plausibility": rel["plausibility"].to_numpy()})
    f = add_alert_layer(f, bundle["alert_cfg"])
    return f, feats, d
