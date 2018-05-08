# -*- coding: utf-8 -*-
"""
Created on Tue May 08 19:06:33 2018

@author: adityac8
"""

# Suppress warnings
import warnings
warnings.simplefilter("ignore")

# Clone the keras_aud library and place the path in ka_path variable
import sys
ka_path="e:/akshita_workspace/cc"
sys.path.insert(0, ka_path)
from keras_aud import aud_feature

wavs_file = 'E:/akshita_workspace/cc/dcase_data/audio/dev/a001_0_30.wav'

X=aud_feature.extract_one(feature_name = 'logmel', wav_file = wavs_file, yaml_file = 'dcase.yaml', dataset = 'dcase_2016')
print X
aud_feature.plot_fig(X)
aud_feature.plot_spec(X)
aud_feature.plot_sim(X)