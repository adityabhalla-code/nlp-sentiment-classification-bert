# sentiment-classification-bert
Sentiment classification using Bert model

## set up the virtual env
```python
# mac os 
python -m venv .venv
source venv/bin/activate
pip install -r requirements.txt
```

## Get data 
```python
python bert_model/get_data.py
# downloads Reviews.csv ( 5lac records)
```
## preprocess data 
```python
python bert_model/preprocessing/preprocess_data.py
# cleans and samples the 60k records and saves preprocessing_data.py
```
## Train model 
```python
python bert_model/train_model.py
```
