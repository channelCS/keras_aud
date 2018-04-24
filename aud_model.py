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

class Feature:    
    def __init__(self,feature):
        self.feature=feature
    def check_dimension(self,n1,dimy):
        """
        Input: Expected Dimension(String), Got Dimension(Integer).
        Output: System Exit if Dimension Mismatch.
        Checks whther the loaded function matches the dimension as
        expected else raises SystemExit. Should be used only with
        custom_check_ftr=True from parameters Else Make
        custom_check_ftr=False.
        """
        if n1 != dimy:
            print "Dimension Mismatch. Expected {} Found {}".format(n1,dimy)
            raise SystemExit
    
def calculate_accuracy(truth,pred):        
	pos,neg=0,0 
	for i in range(0,len(pred)):
		if pred[i] == truth[i]:
			pos = pos+1
		else:
			neg = neg+1
	acc=(float(pos)/float(len(pred)))*100
	return acc

#brock=Feature(feature=feature)
    
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
          
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch,0])
    print(activations)
    return activations

def mat_2d_to_3d(X, agg_num, hop):
    # pad to at least one block
    len_X, n_in = X.shape
    if (len_X < agg_num):
        X = np.concatenate((X, np.zeros((agg_num-len_X, n_in))))
    # agg 2d to 3d
    len_X = len(X)
    i1 = 0
    X3d = []
    while (i1+agg_num <= len_X):
        X3d.append(X[i1:i1+agg_num])
        i1 += hop
    return np.array(X3d)

def mat_3d_to_nd(model, X):
    [batch_num, dimx, dimy]= X.shape 
    if model=='DNN':
        X = X.reshape(batch_num, dimx*dimy)    
    elif model == 'CNN' or model=='CHOU' or model=='CRNN' or model=='FCRNN'or model=='CBRNN':
        X = X.reshape((batch_num,1,dimx,dimy))
    elif model=='RNN':
        X = X.reshape((batch_num,1,dimx*dimy))

    return X
