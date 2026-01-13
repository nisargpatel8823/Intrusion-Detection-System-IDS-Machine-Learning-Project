#!/usr/bin/env bash
set -e
python src/01_eda.py
python src/02_preprocessing.py
python src/03_ml_models.py
python src/04_pytorch_nn.py
python src/05_compare_eval.py
# To test realtime server/client: run server in background then client
# python src/06_realtime_server.py & sleep 1; python src/07_realtime_client.py
