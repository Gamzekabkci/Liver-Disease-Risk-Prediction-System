import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold

from veri_seti import load_and_preprocess_data
from models import get_classifiers, get_ensemble_models, get_ann

# 1. Veri yükle
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data("ILPD.csv")

# 2. Modelleri al
models = {**get_classifiers(), **get_ensemble_models()}
ann = get_ann(X_train.shape[1])
models["ANN"] = ann

# 3. Hold-out değerlendirme
results_holdout = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    sens = recall_score(y_test, y_pred, pos_label=1)  # duyarlılık
    spec = recall_score(y_test, y_pred, pos_label=0)  # özgüllük
    f1 = f1_score(y_test, y_pred)

    results_holdout.append([name, acc, sens, spec, f1])

    # Karışıklık matrisi
    plt.figure(figsize=(4, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=0)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ROC eğrileri için tek figür
plt.figure(figsize=(6, 6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

df_holdout = pd.DataFrame(results_holdout, columns=["Model", "Accuracy", "Sensitivity", "Specificity", "F1-score"])
print("Hold-out sonuçları:\n", df_holdout)

# 4. k-kat çapraz doğrulama (Accuracy, Sensitivity, Specificity, F1-score)
X = np.vstack((X_train, X_test))
y = np.concatenate((y_train, y_test))

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_cv = []

for name, model in models.items():
    acc_scores, sens_scores, spec_scores, f1_scores = [], [], [], []

    for train_idx, test_idx in kfold.split(X, y):
        X_tr, X_val = X[train_idx], X[test_idx]
        y_tr, y_val = y[train_idx], y[test_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        acc_scores.append(accuracy_score(y_val, y_pred))
        sens_scores.append(recall_score(y_val, y_pred, pos_label=1))
        spec_scores.append(recall_score(y_val, y_pred, pos_label=0))
        f1_scores.append(f1_score(y_val, y_pred))

    results_cv.append([name,
                       np.mean(acc_scores),
                       np.mean(sens_scores),
                       np.mean(spec_scores),
                       np.mean(f1_scores)])

df_cv = pd.DataFrame(results_cv, columns=["Model", "CV Accuracy", "CV Sensitivity", "CV Specificity", "CV F1-score"])
print("Cross-validation sonuçları:\n", df_cv)

# 5. ANN için loss grafiği (sklearn MLPClassifier)
ann.fit(X_train, y_train)
plt.figure(figsize=(6, 4))
plt.plot(ann.loss_curve_)
plt.title("ANN Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

from statistical_tests import run_mcnemar

# Hold-out tahminlerini al
y_pred_lr = models["LogisticRegression"].predict(X_test)
y_pred_ab = models["AdaBoost"].predict(X_test)

# McNemar testi çalıştır
run_mcnemar(y_test, y_pred_lr, y_pred_ab, "LogisticRegression", "AdaBoost")

import joblib

# Örneğin AdaBoost en iyi çıktıysa:
best_model = models["AdaBoost"]
best_model.fit(X_train, y_train)

# Model ve scaler kaydet
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ En iyi model ve scaler kaydedildi: best_model.pkl, scaler.pkl")