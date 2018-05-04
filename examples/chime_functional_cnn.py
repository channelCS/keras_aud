# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 03:37:44 2018

@author: adityac8
"""

import warnings
warnings.simplefilter("ignore")

import sys
ka_path="../.."
sys.path.insert(0, ka_path)
from keras_aud import aud_audio, aud_feature
from keras_aud import aud_model, aud_utils

import csv
import cPickle
import numpy as np
import scipy
import time
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from keras.utils import to_categorical
from keras.models import load_model

wav_dev_fd   = ka_path+'/chime_data/audio/dev'
wav_eva_fd   = ka_path+'/chime_data/audio/eva'
dev_fd       = ka_path+'/chime_data/features/dev'
eva_fd       = ka_path+'/chime_data/features/eva'
meta_train_csv  = ka_path+'/chime_data/texts/meta_csvs/development_chunks_refined.csv'
meta_test_csv   = ka_path+'/chime_data/texts/meta_csvs/evaluation_chunks_refined.csv' #eva_csv_path
label_csv       = ka_path+'/chime_data/texts/label_csvs'

labels = [ 'c', 'm', 'f', 'v', 'p', 'b', 'o', 'S' ]

lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }

prep='eval'               # Which mode to use
folds=2                   # Number of folds
#Parameters that are passed to the model.
model_type='Functional'   # Type of model
model='CNN'               # Name of model
feature="logmel"          # Name of feature

dropout1=0.1             # 1st Dropout
act1='relu'              # 1st Activation
act2='sigmoid'              # 2nd Activation
act3='sigmoid'           # 3rd Activation

input_neurons=400      # Number of Neurons
epochs=4              # Number of Epochs
batchsize=128          # Batch Size
num_classes=len(labels) # Number of classes
filter_length=3        # Size of Filter
nb_filter=100          # Number of Filters
#Parameters that are passed to the features.
agg_num=10             # Agg Number(Integer) Number of frames
hop=10                 # Hop Length(Integer)

#aud_audio.extract(feature, wav_dev_fd, dev_fd+'/'+feature,'defaults.yaml')
#aud_audio.extract(feature, wav_eva_fd, eva_fd+'/'+feature,'defaults.yaml')
path='E:/akshita_workspace/chime_home'
aud_utils.unpack_chime_2k16(path,wav_dev_fd,wav_eva_fd,meta_train_csv,meta_test_csv,label_csv)
bre
def GetAllData(fe_fd, csv_file, agg_num, hop):
    """
    Input: Features folder(String), CSV file(String), agg_num(Integer), hop(Integer).
    Output: Loaded features(Numpy Array) and labels(Numpy Array).
    Loads all the features saved as pickle files.
    """
    # read csv
    with open( csv_file, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    
    # init list
    X3d_all = []
    y_all = []
    i=0
    for li in lis:
        # load data
        na = li[1]
        path = fe_fd + '/' + na + '.f'
        info_path = label_csv + '/' + na + '.csv'
        with open( info_path, 'rb') as g:
            reader2 = csv.reader(g)
            lis2 = list(reader2)
        tags = lis2[-2][1]

        y = np.zeros( len(labels) )
        for ch in tags:
            y[ lb_to_id[ch] ] = 1
        try:
            X = cPickle.load( open( path, 'rb' ) )
        except Exception as e:
            print 'Error while parsing',path
            continue
        # reshape data to (n_block, n_time, n_freq)
        i+=1
        X3d = aud_utils.mat_2d_to_3d( X, agg_num, hop )
        X3d_all.append( X3d )
        y_all += [ y ] * len( X3d )
    
    print "Features loaded",i                
    print 'All files loaded successfully'
    # concatenate list to array
    X3d_all = np.concatenate( X3d_all )
    y_all = np.array( y_all )
    
    return X3d_all, y_all



def test(md,csv_file,model):
    # load name of wavs to be classified
    with open( csv_file, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    
    # do classification for each file
    y_pred = []
    te_y = []
    
    for li in lis:
        na = li[1]
        #audio evaluation name
        fe_path = eva_fd + '/'+feature+'/' + na + '.f'
        info_path = label_csv + '/' + na + '.csv'
        with open( info_path, 'rb') as g:
            reader2 = csv.reader(g)
            lis2 = list(reader2)
        tags = lis2[-2][1]

        y = np.zeros( num_classes )
        for ch in tags:
            y[ lb_to_id[ch] ] = 1
        te_y.append(y)
        X0 = cPickle.load( open( fe_path, 'rb' ) )
        X0 = aud_utils.mat_2d_to_3d( X0, agg_num, hop )
        
        X0 = aud_utils.mat_3d_to_nd(model,X0)
    
        # predict
        p_y_pred = md.predict( X0 )
        p_y_pred = np.mean( p_y_pred, axis=0 )     # shape:(n_label)
        y_pred.append(p_y_pred)
    
    return np.array(te_y), np.array(y_pred)



tr_X, tr_y = GetAllData( dev_fd+'/'+feature, meta_train_csv, agg_num, hop )

print(tr_X.shape)
print(tr_y.shape)    
    
tr_X=aud_utils.mat_3d_to_nd(model,tr_X)
print(tr_X.shape)
dimx=tr_X.shape[-2]
dimy=tr_X.shape[-1]

if prep=='dev':
    cross_validation=True
else:
    cross_validation=False
    
miz=aud_model.Functional_Model(input_neurons=input_neurons,cross_validation=cross_validation,dropout1=dropout1,
    act1=act1,act2=act2,act3=act3,nb_filter = nb_filter, filter_length=filter_length,
    num_classes=num_classes,
    model=model,dimx=dimx,dimy=dimy)

np.random.seed(68)
if cross_validation:
    kf = KFold(len(tr_X),folds,shuffle=True,random_state=42)
    results=[]    
    for train_indices, test_indices in kf:
        train_x = [tr_X[ii] for ii in train_indices]
        train_y = [tr_y[ii] for ii in train_indices]
        test_x  = [tr_X[ii] for ii in test_indices]
        test_y  = [tr_y[ii] for ii in test_indices]
        #train_y = to_categorical(train_y,num_classes=len(labels))
        #test_y = to_categorical(test_y,num_classes=len(labels)) 
        
        train_x=np.array(train_x)
        train_y=np.array(train_y)
        test_x=np.array(test_x)
        test_y=np.array(test_y)
        print "Development Mode"

        #get compiled model
        lrmodel=miz.prepare_model()

        if lrmodel is None:
            print "If you have used Dynamic Model, make sure you pass correct parameters"
            raise SystemExit
        #fit the model
        lrmodel.fit(train_x,train_y,batch_size=batchsize,epochs=epochs,verbose=1)
        
        #make prediction
        pred=lrmodel.predict(test_x, batch_size=32)

        pred = [ii.argmax()for ii in pred]
        test_y = [ii.argmax()for ii in test_y]

        results.append(accuracy_score(pred,test_y))
        print accuracy_score(pred,test_y)
        jj=str(set(list(test_y)))
        print "Unique in test_y",jj
    print "Results: " + str( np.array(results).mean() )
else:
    train_x=np.array(tr_X)
    train_y=np.array(tr_y)
    print "Evaluation mode"
    lrmodel=miz.prepare_model()
    #train_y = to_categorical(train_y,num_classes=len(labels))
        
    #fit the model
    lrmodel.fit(train_x,train_y,batch_size=batchsize,epochs=epochs,verbose=1)
    
    truth,pred=test(lrmodel,meta_test_csv,model)

    eer=aud_utils.calculate_eer(truth,pred)
    print "EER %.2f prcnt"%eer
