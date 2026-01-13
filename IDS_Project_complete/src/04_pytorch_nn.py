import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np, joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[1] / 'results' / 'models'
OUT_METRICS = Path(__file__).resolve().parents[1] / 'results' / 'metrics'
OUT_METRICS.mkdir(parents=True, exist_ok=True)

X_train = np.load(MODELS_DIR / 'X_train.npy')
X_test  = np.load(MODELS_DIR / 'X_test.npy')
y_train = np.load(MODELS_DIR / 'y_train.npy')
y_test  = np.load(MODELS_DIR / 'y_test.npy')

input_dim = X_train.shape[1]
device = torch.device('cpu')

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

model = Net(input_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Prepare data
train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float().unsqueeze(1))
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    avg_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}')

# Save model weights and a small wrapper for inference
torch.save(model.state_dict(), MODELS_DIR / 'idsnet.pth')
joblib.dump({'input_dim': input_dim}, MODELS_DIR / 'pytorch_meta.joblib')

# Evaluate
model.eval()
with torch.no_grad():
    preds = model(torch.from_numpy(X_test).float()).numpy().ravel()
y_pred = (preds >= 0.5).astype(int)
rows = [{
    'model': 'PyTorch_NN',
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1': f1_score(y_test, y_pred, zero_division=0)
}]
pd.DataFrame(rows).to_csv(OUT_METRICS / 'pytorch_metrics.csv', index=False)
print('PyTorch training complete. Model saved to', MODELS_DIR)
