# DSTC7

## Introduction

This is the implementation for DSTC7 papers "RAP-Net: Recurrent Attention Pooling Networks for Dialogue Response Selection" and "Learning Multi-Level Information for Dialogue Response Selection by Highway Recurrent Transformer".


## Usage

0. Normalizing the course numbers (only for Advising dataset):
```
python3 prepreprocess_advising.py \
    ../data/task1/advising-train.json \
    ../data/task1/advising-valid.json \
    ../data/task1/advising-train-p.json \
    ../data/task1/advising-valid-p.json
```

1. Build embeddings pickle. Example (It may take about 10 mins on a 16-cores machine):
```
python3 build_embedding.py \
    ../data/task1/advising-train-p.json \
    ../data/task1/advising-valid-p.json \
    ../data/crawl-300d-2M.vec \
    ../data/task1/advising_embeddings.pkl
```

2. Make training data pickle and validation data pickle. Example (It may take a while):
```
python3 make_train_valid_dataset.py \
   ../data/task1/advising_embeddings.pkl \
   ../data/task1/advising-train-p.json \
   ../data/task1/advising-valid-p.json \
   ../data/task1/advising_train.pkl \
   ../data/task1/advising_valid.pkl
```

3. Start training. Example
```
python3 train.py ../models/task1/advising/recurrent_transformer_pool/
```
