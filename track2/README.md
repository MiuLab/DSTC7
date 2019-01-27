# DSTC7 Track2 - Sentence Generation

## Introduction
This is the implementation of the DSTC7 paper "Knowledge-Grounded Response Generation
with Deep Attentional Latent-Variable Model".

## Requirements
Please install the following requirements:
```
- ipdb=0.11
- ruamel_yaml=0.15.46
- python-box=3.2.0
- tqdm=4.25.0
- sumeval=0.1.6
- nltk=3.3.0
- spacy=2.0.16
- faiss=1.3.0
- numpy=1.15.0
- pytorch=0.4.1
- visdom=0.1.8.5
```
and also install `nlg-eval` following the guide in the
[repo](https://github.com/Maluuba/nlg-eval).

## Dataset Preparation
Please refer to the official
[repo](https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling) to download
the data. Then edit `data.yaml` and run `./preprocess.py data.yaml` under the `src`
directory.

## Usage
First, start a visdom server by running `visdom -port PORT_NUMBER -env_path ENV_PATH`.
Then set the training and model parameters in `config.yaml` and run
`./train.py config.yaml` under the `src` directory.
