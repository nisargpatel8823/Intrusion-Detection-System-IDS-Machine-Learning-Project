import joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODELS_DIR = Path(__file__).resolve().parents[1] / 'results' / 'models'
OUT_METRICS = Path(__file__).resolve().parents[1] / 'results' / 'metrics'

X_test = np.load(MODELS_DIR / 'X_test.npy')
y_test = np.load(MODELS_DIR / 'y_test.npy')

rf = joblib.load(MODELS_DIR / 'rf_model.joblib')
svm = joblib.load(MODELS_DIR / 'svm_model.joblib')

# PyTorch model
import torch, torch.nn as nn
from pathlib import Path
from importlib import import_module
meta = joblib.load(MODELS_DIR / 'pytorch_meta.joblib')
input_dim = meta['input_dim']

# Define network (same as training)
class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

nn_model = Net(input_dim)
nn_model.load_state_dict(torch.load(MODELS_DIR / 'idsnet.pth', map_location='cpu'))
nn_model.eval()

models = {'RandomForest': rf, 'SVM': svm, 'PyTorch_NN': nn_model}
rows = []
for name, m in models.items():
    if name == 'PyTorch_NN':
        import torch
        with torch.no_grad():
            preds = m(torch.from_numpy(X_test).float()).numpy().ravel()
        y_pred = (preds >= 0.5).astype(int)
    else:
        y_pred = m.predict(X_test)
    rows.append({
        'model': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    })

pd.DataFrame(rows).to_csv(OUT_METRICS / 'all_models_comparison.csv', index=False)
print('Comparison saved to', OUT_METRICS / 'all_models_comparison.csv')
