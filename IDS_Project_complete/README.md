# IDS Project (Completed scaffold)
This project is prepared for the KDD dataset files `KDDTrain.csv` and `KDDTest.csv` which were included in the uploaded ZIP.

## Structure
```
IDS_Project_complete/
  data/
    KDDTrain.csv
    KDDTest.csv
  src/
    01_eda.py
    02_preprocessing.py
    03_ml_models.py
    04_pytorch_nn.py
    05_compare_eval.py
    06_realtime_server.py
    07_realtime_client.py
  results/
    figures/
    metrics/
    models/
  requirements.txt
  run_all.sh
```
## How to run
1. Create virtualenv and install dependencies:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Run all steps:
```
bash run_all.sh
```
