import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import joblib

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
OUT_DIR = Path(__file__).resolve().parents[1] / 'results' / 'models'
OUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA_DIR / 'KDDTrain.csv')
test  = pd.read_csv(DATA_DIR / 'KDDTest.csv')

label_col = 'label'  # dataset uses 'label' (0 normal, 1 attack)
# Drop any ID-like columns if present
# Identify numeric and categorical columns
num_cols = train.select_dtypes(include=['int64','float64']).columns.tolist()
if label_col in num_cols:
    num_cols.remove(label_col)
cat_cols = [c for c in train.columns if c not in num_cols and c != label_col]

print('Numeric cols:', len(num_cols), 'Categorical cols:', len(cat_cols))

num_pipeline = Pipeline([
    ('impute_num', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('impute_cat', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
], remainder='drop')

# Fit preprocessor on train and save it
X_train = train.drop(columns=[label_col])
y_train = train[label_col].values
preprocessor.fit(X_train)
joblib.dump({'preprocessor': preprocessor, 'num_cols': num_cols, 'cat_cols': cat_cols}, OUT_DIR / 'preprocessor.joblib')
print('Preprocessor saved to', OUT_DIR / 'preprocessor.joblib')

# Transform and save processed numpy arrays for quicker loading
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(test.drop(columns=[label_col]))
import numpy as np
np.save(OUT_DIR / 'X_train.npy', X_train_proc)
np.save(OUT_DIR / 'X_test.npy', X_test_proc)
np.save(OUT_DIR / 'y_train.npy', y_train)
np.save(OUT_DIR / 'y_test.npy', test[label_col].values)

print('Preprocessing complete. Processed arrays saved to results/models.')
