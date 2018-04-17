# Keras Audio Library
[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/channelCS/keras_aud/blob/master/LICENSE) [![dep1](https://img.shields.io/badge/Theano-0.9+-blue.svg)](http://deeplearning.net/software/theano/) [![dep2](https://img.shields.io/badge/Keras-2.1+-blue.svg)](https://keras.io/) 

Neural networks audio toolkit for Keras

import sys
ka_path="e:/akshita_workspace/cc"
sys.path.insert(0, ka_path)
from keras_aud import aud_audio

### Useful Path Definitions:

| Variable        | Description                     |
| :-------------  |:-------------                   |
| `wav_dev_fd`    | Development audio folder        |
| `wav_eva_fd`    | Evaluation audio folder         |
| `dev_fd`        | Development features folder     |
| `eva_fd`        | Evaluation features folder      |

aud_audio.extract('logmel', wav_dev_fd, dev_fd,'example.yaml')
aud_audio.extract('logmel', wav_eva_fd, eva_fd,'example.yaml')

from keras_aud import aud_model

### Model and Feature Parameters

| Variable           | Description              | type       | Accepted values             |
| :-------------     | :-------------           | :--------- | :---------                  |
| `prep`             | mode to use              | `str`      | dev, eval                   |
| `save_model`       | Whether to save model    | `bool`     |                             |
| `model_type`       | Type of model            | `str`      | Dynamic, Functional, Static |
| `model`            | Name of model            | `str`      | DNN, CNN, CRNN, RNN, FCRNN  |
| `modelx`           | Name of model for saving | `str`      | Should end with `.h5`       |
| `feature`          | Name of feature          | `str`      | mel, logmel, cqt, mfcc, zcr |
| **Works only for Functional** | | | |
| `dropout1`         | 1st Dropout              | `float`    |                             |
| `act1`             | 1st Activation           | `str`      |                             |
| `act2`             | 2nd Activation           | `str`      |                             |
| `act3`             | 3rd Activation           | `str`      |                             |
| `act4`             | 4th Activation           | `str`      | Only in case of DNN         |
| **Works for all Models** | | | |
| `input_neurons`    | Number of Neurons        | `int`      |                             |
| `epochs`           | Number of Epochs         | `int`      |                             |
| `batchsize`        | Batch Size               | `int`      |                             |
| `num_classes`      | Number of classes        | `int`      |                             |
| `filter_length`    | Size of Filter           | `int`      |                             |
| `nb_filter`        | Number of Filters        | `int`      |                             |
| **Feature Parameters** | | | |
| `agg_num`          | Number of frames         | `int`      |                             |
| `hop`              | Hop Length               | `int`      |                             |
| `custom_check_ftr` | check for dimensions     | `bool`     | True: know dimension        |
