# DSTC7 Challange
## Prepare Environment
- python 3.6
- pytorch 0.4.0
```
conda install mkl-service
```
## Prepare data

All the data can be found [here](https://drive.google.com/drive/u/2/folders/1JGE4eeelA0QBA7BwYvj89kSClE3f9k65).

You need to first make symbolic link under `AVSD_Jim/data` folder like this.

```
data/
├── word_embedding -> /home/jimlin7777/tmp2/word_embedding/
├── textdata -> /tmp2/DSTC7/AVSD/textdata
├── cache
└── Audio-Visual-Feature -> /tmp2/jimlin7777/DSTC7/AVSD/Audio-Visual-Feature/

word_embedding/
└── glove.6B.300d.txt

```

## Training
```
    python -m AVSD_Jim.src.train --modelType [SimpleModel|ModalAttentionModel] 
```
For more information, use `--help` flag.

## Testing
```
    python -m AVSD_Jim.src.predict [parameter_timestep] 
```
For more information, use `--help` flag.

## Code Structure

```
.
├── data (all data should be placed here)
│   ├── Audio-Visual-Feature -> /tmp2/jimlin7777/DSTC7/AVSD/Audio-Visual-Feature/
│   ├── cache
│   ├── textdata -> /tmp2/DSTC7/AVSD/textdata
│   └── word_embedding -> /home/jimlin7777/tmp2/word_embedding/
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

