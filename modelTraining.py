import os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression as LRCal
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix
)
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, optimizers, Model, Input

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 11

np.random.seed(42)
tf.random.set_seed(42)
os.makedirs("figures", exist_ok=True)
os.makedirs("matrice", exist_ok=True)
os.makedirs("exported_model", exist_ok=True)

def pick_threshold(y_true, p, recall_targets=(0.90, 0.80, 0.70), pos_rate_bounds=None, cap_percentile=0.97):
    prec, rec, thr = precision_recall_curve(y_true, p)
    f1 = 2*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1]+1e-12)
    idx = int(np.nanargmax(f1))
    t = float(thr[idx])
    for r in recall_targets:
        feas = np.where(rec[:-1] >= r)[0]
        if len(feas) > 0:
            f1f = f1[feas]
            t = min(t, float(thr[feas[int(np.nanargmax(f1f))]]))
            break
    cap = float(np.quantile(p, cap_percentile))
    t = min(t, cap)
    if pos_rate_bounds is not None:
        lo, hi = pos_rate_bounds
        pos_rate = (p >= t).mean()
        if pos_rate < lo:
            t = float(np.quantile(p, 1.0 - lo))
        elif pos_rate > hi:
            t = float(np.quantile(p, 1.0 - hi))
    return t

train_df = pd.read_csv("training.csv")
test_df  = pd.read_csv("testing.csv")

y_train_full = train_df["Panic Disorder Diagnosis"].astype(int).values
X_train_df_full = train_df.drop(columns=["Participant ID", "Panic Disorder Diagnosis"]).copy()

y_test = test_df["Panic Disorder Diagnosis"].astype(int).values
X_test_df = test_df.drop(columns=["Participant ID", "Panic Disorder Diagnosis"]).copy()

encoders = {}
for col in X_train_df_full.columns:
    if X_train_df_full[col].dtype == "object":
        le = LabelEncoder()
        X_train_df_full[col] = le.fit_transform(X_train_df_full[col].astype(str))
        encoders[col] = {cls: i for i, cls in enumerate(le.classes_)}
        X_test_df[col] = X_test_df[col].astype(str).map(encoders[col]).fillna(-1).astype(int)

X_train_full = X_train_df_full.values
X_test  = X_test_df.values

X_fit, X_cal, y_fit, y_cal = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
)

class KerasBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_units=(64,), dropout=0.35, l2=1e-4, lr=3e-4, batch_size=64, epochs=60, patience=6, val_split=0.2, verbose=0, random_state=42, lr_patience=4):
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.l2 = l2
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_split = val_split
        self.verbose = verbose
        self.random_state = random_state
        self.lr_patience = lr_patience
        self.model_ = None
        self.threshold_ = 0.5

    def _build(self, input_dim):
        reg = regularizers.l2(self.l2) if self.l2>0 else None
        inp = Input(shape=(input_dim,))
        x = inp
        for u in self.hidden_units:
            x = layers.Dense(u, activation="relu", kernel_regularizer=reg)(x)
            x = layers.Dropout(self.dropout)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        m = Model(inp, out)
        opt = optimizers.Adam(learning_rate=self.lr)
        m.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
        return m

    def fit(self, X, y):
        tf.random.set_seed(self.random_state)
        self.model_ = self._build(X.shape[1])
        es = callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True, verbose=0)
        rl = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=self.lr_patience, min_lr=1e-6, verbose=0)
        self.model_.fit(X, y, validation_split=self.val_split, epochs=self.epochs, batch_size=self.batch_size, callbacks=[es, rl], verbose=self.verbose)
        return self

    def set_threshold(self, t): self.threshold_ = float(t)
    def predict_proba(self, X):
        p = self.model_.predict(X, verbose=0).reshape(-1)
        return np.c_[1.0 - p, p]
    def predict(self, X):
        p = self.model_.predict(X, verbose=0).reshape(-1)
        return (p >= self.threshold_).astype(int)

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Točnost": float(accuracy_score(y_true, y_pred)),
        "Preciznost": float(precision_score(y_true, y_pred, zero_division=0)),
        "Odziv": float(recall_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred)),
        "Uravnotežena točnost": float(balanced_accuracy_score(y_true, y_pred)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "ROC AUC": float(roc_auc_score(y_true, y_prob)),
        "PR AUC": float(average_precision_score(y_true, y_prob)),
    }

base_models = {
    "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state=42)), ("clf", LogisticRegression(max_iter=2000, solver="lbfgs"))]),
    "KNN": Pipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state=42)), ("clf", KNeighborsClassifier(n_neighbors=11))]),
    "RandomForest": Pipeline([("smote", SMOTE(random_state=42)), ("clf", RandomForestClassifier(
        n_estimators=400, max_depth=None, max_features="sqrt", min_samples_leaf=2, bootstrap=True, n_jobs=-1, random_state=42
    ))]),
    "GradientBoosting": Pipeline([("smote", SMOTE(random_state=42)), ("clf", GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=42
    ))]),
    "Keras": Pipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state=42)), ("clf", KerasBinaryClassifier(
        hidden_units=(64,), dropout=0.35, l2=1e-4, lr=3e-4, batch_size=64, epochs=60, patience=6, val_split=0.2, verbose=0, random_state=42
    ))]),
}

detail_rows, fitted_pipes = [], {}
roc_curves, pr_curves, conf_matrices = {}, {}, {}
thresholds, calibrators = {}, {}

print("Treniram i kalibriram pragove...")
for model_name, pipe in base_models.items():
    print(f"- {model_name}")
    pipe.fit(X_fit, y_fit)

    if model_name == "Keras":
        p_cal_raw = pipe.predict_proba(X_cal)[:, 1]
        calib = LRCal(max_iter=1000, solver="lbfgs")
        calib.fit(p_cal_raw.reshape(-1,1), y_cal)
        p_cal = calib.predict_proba(p_cal_raw.reshape(-1,1))[:, 1]
        prev = y_cal.mean()
        t = pick_threshold(y_cal, p_cal, recall_targets=(0.90,0.80,0.70), pos_rate_bounds=(0.5*prev, 1.5*prev))
        pipe.named_steps["clf"].set_threshold(t)
        calibrators[model_name] = calib
        thresholds[model_name] = t
        p_test_raw = pipe.predict_proba(X_test)[:, 1]
        y_prob = calib.predict_proba(p_test_raw.reshape(-1,1))[:, 1]
        y_pred = (y_prob >= t).astype(int)
    else:
        p_cal = pipe.predict_proba(X_cal)[:, 1]
        prev = y_cal.mean()
        t = pick_threshold(y_cal, p_cal, recall_targets=(0.90,0.80,0.70), pos_rate_bounds=(0.5*prev, 1.5*prev))
        thresholds[model_name] = t
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= t).astype(int)

    m = compute_metrics(y_test, y_pred, y_prob)
    detail_rows.append({"Model": model_name, **m})

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    rec, prec, _ = precision_recall_curve(y_test, y_prob)
    roc_curves[model_name] = (fpr, tpr, m["ROC AUC"])
    pr_curves[model_name] = (rec, prec, m["PR AUC"])
    cm = confusion_matrix(y_test, y_pred)
    conf_matrices[model_name] = cm
    fitted_pipes[model_name] = pipe

    plt.figure(figsize=(8,7))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC={m['ROC AUC']:.3f}")
    plt.plot([0,1],[0,1],'--', color="gray", linewidth=1)
    plt.xlabel("Lažno pozitivna stopa"); plt.ylabel("Istinito pozitivna stopa")
    plt.title(f"ROC krivulja – {model_name}")
    plt.legend(loc="lower right", frameon=True); plt.tight_layout()
    plt.savefig(f"figures/ROC_{model_name.replace(' ','_')}.png", dpi=240); plt.close()

    plt.figure(figsize=(8,7))
    plt.plot(rec, prec, linewidth=2, label=f"AP={m['PR AUC']:.3f}")
    plt.xlabel("Odziv"); plt.ylabel("Preciznost")
    plt.title(f"PR krivulja – {model_name}")
    plt.legend(loc="lower left", frameon=True); plt.tight_layout()
    plt.savefig(f"figures/PR_{model_name.replace(' ','_')}.png", dpi=240); plt.close()

    plt.figure(figsize=(5,4))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                     xticklabels=["Negativno","Pozitivno"],
                     yticklabels=["Negativno","Pozitivno"])
    plt.title(f"Matrica zabune – {model_name}")
    plt.xlabel("Predikcija"); plt.ylabel("Stvarno")
    plt.tight_layout(); plt.savefig(f"matrice/konfuzija_test_{model_name.replace(' ','_')}.png", dpi=240); plt.close()

detail_df = pd.DataFrame(detail_rows).round(4)
detail_df.to_csv("metrics_all_models_smote_external.csv", index=False)
with open("selected_thresholds.json", "w", encoding="utf-8") as f:
    json.dump({k: float(v) for k,v in thresholds.items()}, f, indent=2, ensure_ascii=False)

print("\n=== Metrike (eksterni test) ===")
print(detail_df)

plt.figure(figsize=(8,7))
for name, (fpr, tpr, aucv) in roc_curves.items():
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={aucv:.3f})")
plt.plot([0,1],[0,1],'--', color="gray", linewidth=1)
plt.xlabel("Lažno pozitivna stopa"); plt.ylabel("Istinito pozitivna stopa")
plt.title("ROC krivulje – svi modeli (SMOTE)")
plt.legend(loc="lower right", frameon=True); plt.tight_layout()
plt.savefig("figures/ROC_all_models_smote.png", dpi=240); plt.close()

plt.figure(figsize=(8,7))
for name, (rec, prec, apv) in pr_curves.items():
    plt.plot(rec, prec, linewidth=2, label=f"{name} (AP={apv:.3f})")
plt.xlabel("Odziv"); plt.ylabel("Preciznost")
plt.title("PR krivulje – svi modeli (SMOTE)")
plt.legend(loc="lower left", frameon=True); plt.tight_layout()
plt.savefig("figures/PR_all_models_smote.png", dpi=240); plt.close()

metric_order = ["Točnost","Preciznost","Odziv","F1","Uravnotežena točnost","MCC","ROC AUC","PR AUC"]
best_by_metric = {m: detail_df.loc[detail_df[m].idxmax(), "Model"] for m in metric_order}
with open("najbolji_po_metrikama_smote.json", "w", encoding="utf-8") as f:
    json.dump(best_by_metric, f, ensure_ascii=False, indent=2)

trained_keras = fitted_pipes["Keras"]
keras_model = trained_keras.named_steps["clf"].model_
scaler = trained_keras.named_steps["scaler"]
feature_means = scaler.mean_
feature_stds = scaler.scale_
cols = X_train_df_full.columns.tolist()

means_dict = {col: float(m) for col, m in zip(cols, feature_means)}
stds_dict = {col: float(s) for col, s in zip(cols, feature_stds)}
with open("exported_model/feature_means.json", "w", encoding="utf-8") as f:
    json.dump(means_dict, f, indent=2, ensure_ascii=False)
with open("exported_model/feature_stds.json", "w", encoding="utf-8") as f:
    json.dump(stds_dict, f, indent=2, ensure_ascii=False)

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()
with open("exported_model/best_model.tflite", "wb") as f:
    f.write(tflite_model)

meta = {
    "thresholds": {k: float(v) for k,v in thresholds.items()},
    "feature_order": cols,
    "keras_params": {
        "hidden_units": (64,), "dropout": 0.35, "l2": 1e-4, "lr": 3e-4,
        "batch_size": 64, "epochs": 60, "patience": 6, "val_split": 0.2
    }
}
if "Keras" in calibrators:
    coef = float(calibrators["Keras"].coef_.ravel()[0])
    intercept = float(calibrators["Keras"].intercept_.ravel()[0])
    with open("exported_model/calibrator_platt.json", "w", encoding="utf-8") as f:
        json.dump({"coef": coef, "intercept": intercept}, f, indent=2, ensure_ascii=False)
    meta["calibration"] = {"type":"platt","coef":coef,"intercept":intercept}

with open("exported_model/metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("\nArtefakti spremni:")
print("- metrics_all_models_smote_external.csv, selected_thresholds.json, najbolji_po_metrikama_smote.json")
print("- figures/ROC_*.png, PR_*.png, ROC_all_models_smote.png, PR_all_models_smote.png")
print("- matrice/konfuzija_test_*.png")
print("- exported_model/best_model.tflite, feature_means.json, feature_stds.json, metadata.json, calibrator_platt.json (za Keras)")
