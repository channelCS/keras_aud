"""
Created on Sat Apr 08 11:48:18 2018
@author: Akshita Gupta
Email - akshitadvlp@gmail.com

Updated on 15/04/18
@author: Aditya Arora
Email - adityadvlp@gmail.com
"""

import model_description as M
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import time
import csv
import cPickle
import scipy

np.random.seed(1234)

import numpy as np
import sys
import scipy

from keras import backend as K
K.set_image_dim_ordering('th')
   
class Functional_Model:
    def __init__(self,input_neurons,cross_validation,dropout1,
        act1,act2,act3,dimx,dimy,num_classes,
        nb_filter,filter_length,
        model,act4=None,dropout2=None):
        if model==None:
            print "Enter valid model name"
            return
        self.cross_validation=cross_validation
        self.input_neurons=input_neurons
        self.dropout1=dropout1
        self.act1=act1
        self.act2=act2
        self.act3=act3
        self.act4=act4
        self.model=model
        self.nb_filter = nb_filter
        self.filter_length =filter_length
        self.dimx = dimx
        self.dimy = dimy
        self.num_classes=num_classes

    def prepare_model(self):
        if self.model=='DNN':
            lrmodel=M.dnn(num_classes=self.num_classes,input_neurons=self.input_neurons,input_dim=self.dimx*self.dimy,dropout=self.dropout1,act1=self.act1,act2=self.act2,act3=self.act3)
            return lrmodel
        elif self.model=='CNN':
            lrmodel=M.cnn(num_classes=self.num_classes,dropout=self.dropout1,input_neurons=self.input_neurons,act1=self.act1,act2=self.act2,act3=self.act3,dimx = self.dimx,dimy = self.dimy,nb_filter = self.nb_filter,filter_length =self.filter_length)
            return lrmodel
        elif self.model=='RNN':
            lrmodel=M.rnn(num_classes=self.num_classes,input_neurons=self.input_neurons,input_dim=self.dimx*self.dimy)
            return lrmodel
        elif self.model=='CRNN':
            lrmodel=M.cnn_rnn(num_classes=self.num_classes,dropout=self.dropout1,input_neurons=self.input_neurons,act1=self.act1,act2=self.act2,act3=self.act3,nb_filter = self.nb_filter,filter_length =self.filter_length,dimx = self.dimx,dimy = self.dimy)
            return lrmodel
        elif self.model=='FCRNN':
            lrmodel=M.feature_cnn_rnn(num_classes=self.num_classes,dropout=self.dropout1,input_neurons=self.input_neurons,act1=self.act1,act2=self.act2,act3=self.act3,nb_filter = self.nb_filter,filter_length =self.filter_length,dimx = self.dimx,dimy = self.dimy)
            return lrmodel
        elif self.model=='CBRNN':
            lrmodel=M.cbrnn(num_classes=self.num_classes,dropout=self.dropout1,input_neurons=self.input_neurons,act1=self.act1,act2=self.act2,act3=self.act3,nb_filter = self.nb_filter,filter_length =self.filter_length,dimx = self.dimx,dimy = self.dimy)
            return lrmodel
        elif self.model=='MultiCNN':
            lrmodel=M.multi_cnn(num_classes=self.num_classes,dropout=self.dropout1,input_neurons=self.input_neurons,act1=self.act1,act2=self.act2,act3=self.act3,nb_filter = self.nb_filter,filter_length =self.filter_length,dimx = self.dimx,dimy = self.dimy)
            return lrmodel
          
class Static_Model:
    def __init__(self,input_neurons,cross_validation,
        dimx,dimy,num_classes,
        nb_filter,filter_length,
        model):
        if model==None:
            print "Enter valid model name"
            return
        self.cross_validation=cross_validation
        self.input_neurons=input_neurons
        self.model=model
        self.nb_filter = nb_filter
        self.filter_length =filter_length
        self.dimx = dimx
        self.dimy = dimy
        self.num_classes=num_classes

    def prepare_model(self):
        if self.model=='CHOU':
            lrmodel=M.conv_deconv_chou(dimx=self.dimx,dimy=self.dimy,nb_filter=self.nb_filter,num_classes=self.num_classes)
            return lrmodel
                
class Dynamic_Model:
    def __init__(self,model,num_classes,dimx,dimy,acts,**kwargs):
        if model==None:
            print "Enter valid model name"
            return
        self.model=model
        self.num_classes=num_classes
        self.dimx = dimx
        self.dimy = dimy
        self.acts = acts
        self.kwargs=kwargs
    def prepare_model(self):
        try:
            if self.model=='DNN':
                lrmodel=M.dnn_dynamic(num_classes = self.num_classes,
                                      input_dim   = self.dimx*self.dimy,
                                      acts        = self.acts,
                                      kwargs      = self.kwargs)
            elif self.model=='CNN':
                lrmodel=M.cnn_dynamic(num_classes = self.num_classes,
                                      dimx        = self.dimx,
                                      dimy        = self.dimy,
                                      acts        = self.acts,
                                      kwargs      = self.kwargs)
            elif self.model=='CBRNN':
                lrmodel=M.cbrnn_dynamic(num_classes = self.num_classes,
                                        dimx        = self.dimx,
                                        dimy        = self.dimy,
                                        acts        = self.acts,
                                        kwargs      = self.kwargs)
            return lrmodel
        except Exception as e:
            print(e)
          
