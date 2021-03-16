@echo off 
python -m venv nlp-semantic-env
.\nlp-semantic-env\Scripts\activate  
pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html 
pip install -r requirements.txt  
python .\predictor.py