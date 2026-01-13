import joblib, numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[1] / 'results' / 'models'
OUT_METRICS = Path(__file__).resolve().parents[1] / 'results' / 'metrics'
OUT_METRICS.mkdir(parents=True, exist_ok=True)

data = joblib.load(MODELS_DIR / 'preprocessor.joblib')
preprocessor = data['preprocessor']

X_train = np.load(MODELS_DIR / 'X_train.npy')
X_test  = np.load(MODELS_DIR / 'X_test.npy')
y_train = np.load(MODELS_DIR / 'y_train.npy')
y_test  = np.load(MODELS_DIR / 'y_test.npy')

print('Training RandomForest...')
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, MODELS_DIR / 'rf_model.joblib')

print('Training SVM (may be slow)...')
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train)
joblib.dump(svm, MODELS_DIR / 'svm_model.joblib')

# Evaluate
models = {'RandomForest': rf, 'SVM': svm}
rows = []
for name, m in models.items():
    y_pred = m.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    rows.append({'model': name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})
    # save classification report & confusion matrix
    cr = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(cr).T.to_csv(OUT_METRICS / f'{name}_classification_report.csv')
    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm, index=['true_0','true_1'], columns=['pred_0','pred_1']).to_csv(OUT_METRICS / f'{name}_confusion_matrix.csv')

pd.DataFrame(rows).to_csv(OUT_METRICS / 'model_metrics_summary.csv', index=False)
print('Model training & evaluation finished. Models saved to', MODELS_DIR)
