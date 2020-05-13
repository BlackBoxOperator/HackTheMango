# Hack The Mango

## approaches

### Deep Learning

- try multiple state-of-art [models](https://paperswithcode.com/task/image-classification?fbclid=IwAR0ZilWObSB00Q5k6RLbnYEKuTFe2xNz_su44IVMq-ELh_MRV4Savn5kJ9E)
    - [effecientNet](https://github.com/qubvel/efficientnet)
    - CNN + RNN
        - [1](https://github.com/vinayakumarr/CNN-RNN)
        - [2](https://github.com/tvavlad123/cnn-rnn-lstm-image-recognition)
        - [3](https://github.com/ShadyF/cnn-rnn-classifier)
- add [noise](https://machinelearningmastery.com/how-to-improve-deep-learning-model-robustness-by-adding-noise/?fbclid=IwAR3GGa9xD57gPrNz016tp3Go3EZzIycFmExR32CFB1P-3TvgFmFTZfNz6Ho)
- object detection (DL approach)
    - after CONV, maxPooling
    - after Dense 128

### Traditional CV Rule Based

- object detection & segmentation (Traditional approach)
    [1](https://github.com/DtCarrot/fruit-segmentation-mask-rcnn)
    [2](https://github.com/nicolaihaeni/MinneApple)
- background removing
- color feature
- morphic feature
- fruit blob detection
    [1](https://www.itread01.com/content/1546661369.html)
    [2](http://brightguo.com/blob-detection-using-opencv-python-c/)
    [3](https://kknews.cc/zh-tw/news/83zkbne.html)

### Other strategies

- xgboost, SVM, J48
- feature selection
- postprocessing (remove miss pred)
- preprocessing (add/remove noising)
- augmentation
- image size
- transfer learning


## resources

kaggle fruit [dataset](https://www.kaggle.com/mbkinaci/fruit-images-for-object-detection) (enhance the coco model of [imageAI](http://imageai.org/)?)

kaggle [fruit360](https://www.kaggle.com/moltean/fruits/kernels)
