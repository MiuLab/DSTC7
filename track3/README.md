# DSTC7 Challange
## Prepare Environment
- python 3.6
- pytorch 0.4.0
```
conda install mkl-service
```
## Prepare data

All the data can be found in the ofiicial repository [here](https://github.com/hudaAlamri/DSTC7-Audio-Visual-Scene-Aware-Dialog-AVSD-Challenge).

You need to first make symbolic link under `data` folder like this.

```
data/
├── word_embedding 
├── textdata
├── cache
└── Audio-Visual-Feature 

word_embedding/
└── glove.6B.300d.txt

```

## Training
```
    python -m src.train --modelType [SimpleModel|....] 
```
For more information, use `--help` flag.
A simple example can be found in `train_simple.sh`

## Testing
```
    python -m src.predict [parameter_timestep] 
```
For more information, use `--help` flag.

## Code Structure

```
.
├── data (all data should be placed here)
│   ├── Audio-Visual-Feature 
│   ├── cache
│   ├── textdata 
│   └── word_embedding 
├── output (all model output would be place here)
│   ├── generate (after predicting, model prediced result will be place here as a JSON file)
│   ├── log (after training, training losses and validation losses will be placed here)
│   ├── metrics (after predicting, automatic evaluation metrics would be placed here)
│   ├── parameter (after training, model parameters would be placed here)
│   └── visualization 
└── src
    ├── components (Layers that can be reuse)
    ├── dataset (handle the dataset)
    ├── log (log configuration file)
    ├── model (all models are here)
    │   ├── components
    │   └── net
    ├── statistics (plot things or compute statistics information)
    └── util 
```

