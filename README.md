# Hack The Mango

## prepare the data

To get the cp1 data, you must cd to `cp1_data` directory, and follow the `README.md` in it.

I have already do the naive object detection on the data set. You can download it via executing `bash fetch_obj_deted.sh`, but I don't think object detected data is useful now, so you can skip it.

## execute the code

If you want to train the model, go into the models directory and choose a model file such as `code/NaiveCNN/models/naive_cnn.py`.

```
cd code/NaiveCNN/models
python naive_cnn.py
```

Then you will get the `.h5` file, you can use the predict script to do summary.

```
cd code/NaiveCNN
python naive_cnn_predict.py models/naive_cnn.h5
```

## Ensemble resluts

strip the file name in result

```
python Kaggle-Ensemble-Guide/src/strip_file_name.py test_before1.csv test_after1.csv
```

Ensemble multiple result

```
python Kaggle-Ensemble-Guide/src/kaggle_vote.py test_after*.csv test_ensemble.csv
```
