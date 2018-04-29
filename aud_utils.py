# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 15:26:13 2018

@author: @author: Aditya Arora
Email - adityadvlp@gmail.com
"""
from keras import backend as K
import numpy as np

def check_dimension(feature,dimy,yaml_file):
    """
    Input: Got Dimension(Integer), YAML file from which feature was extracted
    to get Expected Dimension(Integer).
    Output: System Exit if Dimension Mismatch.
    Checks whther the loaded function matches the dimension as
    expected else raises SystemExit. Should be used only with
    custom_check_ftr=True from parameters Else Make
    custom_check_ftr=False.
    """
    if feature in ['mel','logmel']:
        find='n_mels'
    elif feature in ['cqt','spectralcentroid']:
        find='n_mels'
    
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
