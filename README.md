# Keras Audio Library
[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/channelCS/keras_aud/blob/master/LICENSE) [![dep1](https://img.shields.io/badge/Theano-0.9+-blue.svg)](http://deeplearning.net/software/theano/) [![dep2](https://img.shields.io/badge/Keras-2.1+-blue.svg)](https://keras.io/) 

Neural networks audio toolkit for Keras.

## Running the library
Clone the repo using 
```
git clone https://github.com/akshitac8/keras_aud.git
```
and add the **path** in `ka_path`. 

```python
import sys
ka_path="path/to/keras_aud"
sys.path.insert(0, ka_path)
```
## Extracting audio features

We can extract a variety of features. See features list.

```python
from keras_aud import aud_audio
aud_audio.extract(feature_name, 'path/to/wavs', 'desired/features/path','yaml_file')
```

## Making a functional model

We shall now make a functional CNN `model='CNN'` model in which we pass `train_x` having shape as `(batch_size,1,dimx,dimy)` and `train_y` with shape as `(batch_size,num_classes)`. The model contains `nb_filter = 100, filter_length=5`. Three activations are used `act1='relu',act2='relu'`. Each layer is followed by a `dropout1=0.1`. We then add a dense layer with `act3='sigmoid', input_neurons=400` and another having `output shape=num_classes`in the end.

```python
from keras_aud import aud_model
miz=aud_model.Functional_Model(model='CNN',input_neurons=400,dropout1=0.1,
    act1='relu',act2='relu',act3=act3,nb_filter = 100, filter_length=5,
    num_classes=15,dimx=10,dimy=40)

```
This model can be used for making predictions.

This returns us a **class** object. We can then `compile` the model using

```python
lrmodel=miz.prepare_model()
lrmodel.fit(train_x,train_y,batch_size=batchsize,epochs=epochs,verbose=1)
```

## Extracting a single feature
We use `aud_feature` to extract features a single `wav` file. We need to provide:
- A path to the audio in `.wav` format
- A yaml file containing feature description.
- The feature we want to extract

Extracting features is easy. Simply pass the required parameters to `extract_one` function.

```python
from keras_aud import aud_feature
wavs_file = 'wavs/sample.wav'
aud_feature.extract_one(feature_name = 'mel', wav_file = wavs_file, yaml_file = 'defaults.yaml')
```

### Plotting the feature

```python
feature.plot_fig(X)
```
<img src="./examples/images/plot.png" />

### Plotting the specgram

```python
feature.plot_spec(X)
```
<img src="./examples/images/spec.png" />
