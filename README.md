# SentimentExtraction from Images and Videos
Digital image processing project to extract human sentiment from videos and images
data used https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

## Requirements

* tensorflow
* keras
* numpy

## Data setup
* Download data from link provided
* Extract file fer2013.csv and run
```
python conv.py --cvs_path path/to/fer2013.csv
```

## Training

To train the model run:

```train
python train.py --data_path path/to/data/folder
```

## Demo

To evaluate the model run

```eval
python detect_sentiment.py
```

