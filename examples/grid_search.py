# -*- coding: utf-8 -*-
"""
Created on Tue May 08 20:37:15 2018

@author: akshitac8
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
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

# This is where all audio files reside and features will be extracted
audio_ftr_path='E:/akshita_workspace/git_x'

# We now tell the paths for audio, features and texts.
wav_dev_fd   = audio_ftr_path+'/dcase_data/audio/dev'
wav_eva_fd   = audio_ftr_path+'/dcase_data/audio/eva'
dev_fd       = audio_ftr_path+'/dcase_data/features/dev'
eva_fd       = audio_ftr_path+'/dcase_data/features/eva'
label_csv    = audio_ftr_path+'/Summaries/utils/dcase_data/texts/dev/meta.txt'
txt_eva_path = audio_ftr_path+'/Summaries/utils/dcase_data/texts/eva/test.txt'
new_p        = audio_ftr_path+'/Summaries/utils/dcase_data/texts/eva/evaluate.txt'

#aud_audio.extract('logmel', wav_dev_fd, dev_fd,'example.yaml')
#aud_audio.extract('logmel', wav_eva_fd, eva_fd,'example.yaml')

labels = [ 'bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path', 'grocery_store', 'home', 'beach', 
            'library', 'metro_station', 'office', 'residential_area', 'train', 'tram', 'park' ]
lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }



prep='eval'               # Which mode to use
folds=4                   # Number of folds
#Parameters that are passed to the model.
model_type='Functional'   # Type of model
model='CNN'               # Name of model
feature="logmel"          # Name of feature

dropout1=0.1             # 1st Dropout
act1='relu'              # 1st Activation
act2='sigmoid'              # 2nd Activation
act3='softmax'           # 3rd Activation

input_neurons=400      # Number of Neurons
epochs=10              # Number of Epochs
batchsize=128          # Batch Size
num_classes=15         # Number of classes
filter_length=3        # Size of Filter
nb_filter=100          # Number of Filters
#Parameters that are passed to the features.
agg_num=10             # Agg Number(Integer) Number of frames
hop=10                 # Hop Length(Integer)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
def seq_cnn():
    md=Sequential()
    md.add(Conv2D(input_shape=(1,dimx,dimy),filters=nb_filter,kernel_size=filter_length,data_format='channels_first',padding='same',activation=act1))

    md.add(MaxPooling2D(pool_size=(2,2)))
    md.add(Flatten())
    md.add(Dense(input_neurons, activation=act2,name='wrap'))
    md.add(Dropout(0.1))
    md.add(Dense(num_classes,activation=act3,name='score'))
    
    md.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])
    return md

from keras.models import Model
from keras.layers import Input
def func_cnn():
    inpx = Input(shape=(1,dimx,dimy))
    x=Conv2D(filters=nb_filter,kernel_size=filter_length,data_format='channels_first',padding='same',activation=act1)(inpx)

    x=MaxPooling2D(pool_size=(2,2))(x)
    x=Flatten()(x)
    x=Dense(input_neurons, activation=act2,name='wrap')(x)
    x=Dropout(0.1)(x)
    score=Dense(num_classes,activation=act3,name='score')(x)
    md = Model([inpx],score)

    md.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])
    return md

def GetAllData(fe_fd, csv_file):
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
        [na, lb] = li[0].split('\t')
        na = na.split('/')[1][0:-4]
        path = fe_fd + '/' + na + '.f'
        try:
            X = cPickle.load( open( path, 'rb' ) )
        except Exception as e:
            print 'Error while parsing',path
            continue
        # reshape data to (n_block, n_time, n_freq)
        i+=1
        X3d = aud_utils.mat_2d_to_3d( X, agg_num, hop )
        X3d_all.append( X3d )
        y_all += [ lb_to_id[lb] ] * len( X3d )
    
    print "Features loaded",i                
    print 'All files loaded successfully'
    # concatenate list to array
    X3d_all = np.concatenate( X3d_all )
    y_all = np.array( y_all )
    
    return X3d_all, y_all



def test(md,csv_file):
    # load name of wavs to be classified
    with open( csv_file, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    
    # do classification for each file
    names = []
    pred_lbs = []
    
    for li in lis:
        names.append( li[0] )
        na = li[0][6:-4]
        #audio evaluation name
        fe_path = eva_fd + '/' + feature + '/' + na + '.f'
        X0 = cPickle.load( open( fe_path, 'rb' ) )
        X0 = aud_utils.mat_2d_to_3d( X0, agg_num, hop )
        
        X0 = aud_utils.mat_3d_to_nd(model,X0)
    
        # predict
        p_y_preds = md.predict(X0)        # probability, size: (n_block,label)
        preds = np.argmax( p_y_preds, axis=-1 )     # size: (n_block)
        b = scipy.stats.mode(preds)
        pred = int( b[0] )
        pred_lbs.append( id_to_lb[ pred ] )
    
    pred = []    
    # write out result
    for i1 in xrange( len( names ) ):
        fname = names[i1] + '\t' + pred_lbs[i1] + '\n' 
        pred.append(fname)
        
    print 'write out finished!'
    truth = open(new_p,'r').readlines()
    pred = [i.split('\t')[1].split('\n')[0]for i in pred]
    truth = [i.split('\t')[1]for i in truth]
    pred.sort()
    truth.sort()
    return truth,pred



tr_X, tr_y = GetAllData( dev_fd+'/'+feature, label_csv )

print(tr_X.shape)
print(tr_y.shape)    
    
dimx=tr_X.shape[-2]
dimy=tr_X.shape[-1]
tr_X=aud_utils.mat_3d_to_nd(model,tr_X)
print(tr_X.shape)

if prep=='dev':
    cross_validation=True
else:
    cross_validation=False
    
miz=aud_model.Functional_Model(input_neurons=input_neurons,cross_validation=cross_validation,dropout1=dropout1,
    act1=act1,act2=act2,act3=act3,nb_filter = nb_filter, filter_length=filter_length,
    num_classes=num_classes,
    model=model,dimx=dimx,dimy=dimy)

np.random.seed(68)
bre
lrmodel = KerasClassifier(build_fn=func_cnn, verbose=1)

train_x=np.array(tr_X)
train_y=np.array(tr_y)
train_y = to_categorical(train_y,num_classes=len(labels))

# define the grid search parameters
batch_size = [10, 20]
epochs = [10, 20]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=lrmodel, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(train_x,train_y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
