import socket, json, pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
HOST='127.0.0.1'
PORT=9999

# Read a sample from test dataset and send as JSON
df = pd.read_csv(DATA_DIR / 'KDDTest.csv')
sample = df.drop(columns=['label']).iloc[0].to_dict()  # send first row (without label)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall((json.dumps(sample) + '\n').encode())
    resp = s.recv(4096)
    print('Server response:', resp.decode())
