import socket, json, joblib, pandas as pd
from pathlib import Path
MODELS_DIR = Path(__file__).resolve().parents[1] / 'results' / 'models'

# Load preprocessor and model (RandomForest pipeline: preprocessor saved separately)
data = joblib.load(MODELS_DIR / 'preprocessor.joblib')
preprocessor = data['preprocessor']
rf = joblib.load(MODELS_DIR / 'rf_model.joblib')

HOST='127.0.0.1'
PORT=9999

def predict_single(sample_dict):
    # sample_dict: mapping feature -> value (raw)
    df = pd.DataFrame([sample_dict])
    # Ensure columns order by using training preprocessor inputs if needed
    X_proc = preprocessor.transform(df)
    pred = rf.predict(X_proc)[0]
    proba = rf.predict_proba(X_proc).max(axis=1)[0] if hasattr(rf, 'predict_proba') else None
    return {'prediction': int(pred), 'score': float(proba) if proba is not None else None}

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    print('Server listening on {}:{}'.format(HOST, PORT))
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        buffer = b''
        while True:
            data = conn.recv(4096)
            if not data:
                break
            buffer += data
            if b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                try:
                    req = json.loads(line.decode())
                    res = predict_single(req)
                except Exception as e:
                    res = {'error': str(e)}
                conn.sendall((json.dumps(res) + '\n').encode())
