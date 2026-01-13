import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
OUT_DIR = Path(__file__).resolve().parents[1] / 'results' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

train_path = DATA_DIR / 'KDDTrain.csv'
test_path  = DATA_DIR / 'KDDTest.csv'

print('Loading datasets...')
train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)
print('Train shape:', train.shape)
print('Test shape:', test.shape)

print('\nColumns:') 
print(train.columns.tolist())

# Basic stats
train.describe(include='all').to_csv(Path(__file__).resolve().parents[1] / 'results' / 'metrics' / 'train_describe.csv')

# Label distribution
if 'label' in train.columns:
    dist = train['label'].value_counts().sort_index()
    dist.plot(kind='bar')
    plt.title('Label counts (train)')
    plt.xlabel('label')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'label_counts_train.png')
    plt.clf()

# Attack type distribution (if exists)
if 'attack_type' in train.columns:
    train['attack_type'].value_counts().head(20).plot(kind='bar', figsize=(10,4))
    plt.title('Top attack types (train)')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'attack_types_train.png')
    plt.clf()

# Correlation heatmap on numeric columns (sampled if large)
num = train.select_dtypes(include='number')
if num.shape[1] > 1:
    corr = num.corr().abs()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap='viridis')
    plt.title('Numeric feature correlations (train)')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'corr_heatmap.png')
    plt.clf()

print('EDA finished. Figures saved to', OUT_DIR)
