"""
Created on Sat Apr 08 11:48:18 2018

@author: Akshita Gupta
Email - akshitadvlp@gmail.com
"""

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import BatchNormalization, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers import Embedding, LSTM, Reshape, Bidirectional, TimeDistributed, Permute
from keras.models import load_model
from keras import backend as K
import numpy as np

############################# Keras shape #############################
def kr(t,m=None):
    if m is None:
        return t._keras_shape
    else:
        return t._keras_shape[m]
    
########################### BASIC DNN #################################
def dnn(input_neurons,input_dim,dropout,num_classes,act1=None,act2=None,act3=None,act4='softmax'):
    print "Activation 1 {} 2 {} 3 {} 4 {}".format(act1,act2,act3,act4)
    print "Model DNN"
    inpx = Input(shape=(input_dim,),name = 'inpx')
    x = Dense(input_neurons, activation=act1)(inpx)
    x= Dropout(dropout)(x)
    x= Dense(input_neurons, activation=act2)(x)
    x= Dropout(dropout)(x)
    x = Dense(input_neurons, activation=act3)(x)
    x= Dropout(dropout)(x)
    score = Dense(num_classes, activation=act4)(x)
    model = Model([inpx],score)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    
    return model

########################### BASIC CNN #################################

def cnn(input_neurons,dimx,dimy,dropout,nb_filter,
                         filter_length,num_classes,pool_size=(3,3),act1=None,act2=None,act3=None):
    print "Activation 1 {} 2 {} 3 {}".format(act1,act2,act3)
    print "Model CNN"
    inpx = Input(shape=(1,dimx,dimy),name='inpx')
    
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation=act1)(inpx)

    hx = MaxPooling2D(pool_size=pool_size)(x)
    h = Flatten()(hx)
    wrap = Dense(input_neurons, activation=act2,name='wrap')(h)
    wrap= Dropout(dropout)(wrap)
    score = Dense(num_classes,activation=act3,name='score')(wrap)
    
    model = Model([inpx],score)
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
    
    return model

########################### BASIC RNN #################################
def rnn(input_neurons,input_dim,num_classes):
    main_input = Input(shape=(1,input_dim), name='main_input')
    x = LSTM(32)(main_input)

    # We stack a deep densely-connected network on top
    x = Dense(input_neurons, activation='relu')(x)
    x = Dense(input_neurons, activation='relu')(x)
    x = Dense(input_neurons, activation='relu')(x)
    
    # And finally we add the main logistic regression layer
    main_output = Dense(num_classes, activation='sigmoid', name='main_output')(x)
    model = Model(inputs=main_input, outputs=main_output)
    model.summary()
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    return model

########################### BASIC CRNN #################################

def cnn_rnn(input_neurons,dimx,dimy,num_classes,nb_filter,filter_length,act1,act2,act3,pool_size=(2,2),dropout=0.1):
    
    main_input = Input(shape=(1,dimx,dimy))
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation=act1)(main_input)
    hx = MaxPooling2D(pool_size=pool_size)(x)
    wrap= Dropout(dropout)(hx)
    x = Permute((2,1,3))(wrap)
    a,b,c,d= kr(x)
    x = Reshape((b*d,c))(x)
    x = LSTM(32)(x)
    wrap= Dropout(dropout)(x)
    x = Dense(input_neurons, activation=act3)(wrap)
    main_output = Dense(num_classes, activation='softmax', name='main_output')(wrap)
    model = Model(inputs=main_input, outputs=main_output)
    model.summary()
    model.compile(loss='categorical_crossentropy',
			  optimizer='adadelta',
			  metrics=['accuracy'])
    
    return model
########################### BASIC FCRNN #################################
def feature_cnn_rnn(input_neurons,dimx,dimy,num_classes,nb_filter,filter_length,act1,act2,act3,pool_size=(2,2),dropout=0.1):
    #Extract features using CNN and pass them as a input to LSTM
    main_input = Input(shape=(1,dimx,dimy))
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation=act1)(main_input)
    hx = MaxPooling2D(pool_size=pool_size)(x)
    wrap= Dropout(dropout)(hx)
    h = Flatten()(wrap)
    wrap = Dense(input_neurons, activation=act2,name='wrap')(h)

    score = Dense(num_classes,activation=act3,name='score')(wrap)
    model = Model([main_input],score)
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
    
    return model

############################# BASIC CBRNN #############################
def cbrnn(input_neurons,dimx,dimy,num_classes,nb_filter,filter_length,act1,act2,act3,pool_size=(2,2),dropout=0.1):
    #CNN with biderectional lstm
    print "CBRNN"
    main_input = Input(shape=(1,dimx,dimy))
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation='tanh')(main_input)
    hx = MaxPooling2D(pool_size=pool_size)(x)
    wrap= Dropout(dropout)(hx)
    x = Permute((2,1,3))(wrap)
    a,b,c,d= kr(x)
    x = Reshape((b*d,c))(x) 
    w = Bidirectional(LSTM(32))(x)
    wrap= Dropout(dropout)(w)
    main_output = Dense(num_classes, activation='sigmoid', name='main_output')(wrap)
    model = Model(inputs=main_input, outputs=main_output)
    model.summary()
    model.compile(loss='categorical_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])
    
    return model

########################################### DYNAMIC MODELS ###########################################

########################### DYNAMIC DNN #################################
def dnn_dynamic(input_neurons,input_dim,num_classes,layers=0,acts=[],drops=[]):
    layers2 = layers-1
    last    = acts.pop()
    last2   = acts.pop()
    dropout = drops.pop()
    if not np.all([len(drops)==layers2,len(acts)==layers2]):
        print "Layers Mismatch"
        return False
    inpx = Input(shape=(input_dim,),name='inpx')
    inpx2=inpx
    for i in range(layers2):
        x = Dense(input_neurons,activation=acts[i])(inpx)
        inpx=   Dropout(drops[i])(x)

    wrap = Dense(input_neurons, activation=last2,name='wrap')(inpx)
    wrap= Dropout(dropout)(wrap)
    score = Dense(num_classes,activation=last,name='score')(wrap)
    
    model = Model(inpx2,score)
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    return model

########################### DYNAMIC CNN #################################

def cnn_dynamic(input_neurons,dimx,dimy,nb_filter,
                         filter_length,num_classes,layers=0,pools=[],acts=[],drops=[],bn=False):
    layers2 = layers-1
    last    = acts.pop()
    last2   = acts.pop()
    dropout = drops.pop()
    if not np.all([len(pools)==layers2,len(drops)==layers2,len(acts)==layers2]):
        print "Layers Mismatch"
        return False
    inpx = Input(shape=(1,dimx,dimy),name='inpx')
    inpx2=inpx
    for i in range(layers2):
        x = Conv2D(filters=nb_filter,
                   kernel_size=filter_length,
                   data_format='channels_first',
                   padding='same',
                   activation=acts[i])(inpx)
        if bn:
            x=BatchNormalization()(x)
        hx = MaxPooling2D(pool_size=pools[i])(x)
        inpx=   Dropout(drops[i])(hx)

    h = Flatten()(inpx)
    wrap = Dense(input_neurons, activation=last2,name='wrap')(h)
    wrap= Dropout(dropout)(wrap)
    score = Dense(num_classes,activation=last,name='score')(wrap)
    
    model = Model(inpx2,score)
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    return model



#############################  model.conv_deconv #################################

def conv_deconv_chou(dimx,dimy,nb_filter,num_classes):
    inpx = Input(shape=(1,dimx,dimy),name='inpx')
    ### 1st Convolution Layer #############
    x = Conv2D(filters=nb_filter,
               kernel_size=3,
               data_format='channels_first',
               padding='same',
               activation='relu')(inpx)
    hx = MaxPooling2D(pool_size=(4,1))(x)
    hx = ZeroPadding2D(padding=(2, 1))(hx)
    ### 2nd Convolution Layer #############
    x = Conv2D(filters=nb_filter,
               kernel_size=6,
               data_format='channels_first',
               padding='same',
               activation='tanh')(hx)
    hx = MaxPooling2D(pool_size=(4,1))(x)
    hx = ZeroPadding2D(padding=(2, 1))(hx)
    ### 3rd Convolution Layer #############
    x = Conv2D(filters=nb_filter,
               kernel_size=6,
               data_format='channels_first',
               padding='same',
               activation='tanh')(hx)
    hx = MaxPooling2D(pool_size=(4,1))(x)
    hx = ZeroPadding2D(padding=(2, 1))(hx)
    ###   DeConvolution Layer #############
    """
    hx=Conv2DTranspose(filters=nb_filter,
               kernel_size=3,
               data_format='channels_first',
               padding='same',
               strides=(4,1))(hx)
    hx=Conv2DTranspose(filters=nb_filter,
               kernel_size=3,
               data_format='channels_first',
               padding='same',
               strides=(4,1))(hx)
    score = Conv2D(filters=num_classes,
               kernel_size=6,
               data_format='channels_first',
               padding='same',
               activation='sigmoid')(hx)
    """
    
    pooling=GlobalAveragePooling2D()(hx)
    dense = Dense(1024)(pooling)    
    dense = Dense(1024)(dense)    
    score = Dense(num_classes,activation='softmax')(dense)
    
    model = Model([inpx],score)
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    return model

def conv_deconv(input_neurons,dimx,dimy,dropout,nb_filter,
                         filter_length,num_classes,pool_size=(3,3),act1=None,act2=None,act3=None):
    print "Activation 1 {} 2 {} 3 {}".format(act1,act2,act3)
    print "Model CNN with Dropout"
    inpx = Input(shape=(1,dimx,dimy),name='inpx')
    
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation=act1)(inpx)

    hx = MaxPooling2D(pool_size=pool_size)(x)
    h = Flatten()(hx)
    wrap = Dense(input_neurons, activation=act2,name='wrap')(h)
    wrap= Dropout(dropout)(wrap)
    score = Dense(num_classes,activation=act3,name='score')(wrap)
    
    model = Model([inpx],score)
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    return model
