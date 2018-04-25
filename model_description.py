"""
Created on Sat Apr 08 11:48:18 2018

@author: Akshita Gupta
Email - akshitadvlp@gmail.com
"""

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import BatchNormalization, ZeroPadding2D
from keras.layers import Embedding, LSTM, GRU, Reshape, Bidirectional, TimeDistributed, Permute
from keras.models import load_model
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
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
    w = Bidirectional(LSTM(32,return_sequences=False))(x)
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
def dnn_dynamic(num_classes,input_dim,acts,**kwargs):
    input_neurons = kwargs['kwargs'].get('input_neurons',200)
    drops         = kwargs['kwargs'].get('drops',[])
    dnn_layers    = kwargs['kwargs'].get('dnn_layers',1)
    last_act      = kwargs['kwargs'].get('last_act','softmax')
    end_dense = kwargs['kwargs'].get('end_dense',{})

    
    if not np.all([len(acts)==dnn_layers]):
        print "Layers Mismatch"
        return False
    x = Input(shape=(input_dim,),name='inpx')
    inpx = x
    for i in range(dnn_layers):
        x = Dense(input_neurons,activation=acts[i])(inpx)
        if drops != []:
            x = Dropout(drops[i])(x)

    if end_dense != {}:
        x = Dense(end_dense['input_neurons'], activation=end_dense['activation'],name='wrap')(x)
        try:
            x = Dropout(end_dense['dropout'])(x)
        except:
            pass
    score = Dense(num_classes,activation=last_act,name='score')(x)
    
    model = Model(inpx,score)
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    return model

########################### DYNAMIC CNN #################################

def cnn_dynamic(num_classes,dimx,dimy,acts,**kwargs):
    cnn_layers = kwargs['kwargs'].get('cnn_layers',1)
    nb_filter     = kwargs['kwargs'].get('nb_filter',[])
    filter_length = kwargs['kwargs'].get('filter_length',[])

    pools     = kwargs['kwargs'].get('pools',[])
    drops     = kwargs['kwargs'].get('drops',[])
    bn        = kwargs['kwargs'].get('batch_norm',False)
    end_dense = kwargs['kwargs'].get('end_dense',{})
    last_act  = kwargs['kwargs'].get('last_act','softmax')

    if not np.all([len(acts)==cnn_layers,len(nb_filter)==cnn_layers,len(filter_length)==cnn_layers]):
        print "Layers Mismatch"
        return False
    x = Input(shape=(1,dimx,dimy),name='inpx')
    inpx = x
    for i in range(cnn_layers):
        x = Conv2D(filters=nb_filter[i],
                   kernel_size=filter_length[i],
                   data_format='channels_first',
                   padding='same',
                   activation=acts[i])(x)
        if bn:
            x=BatchNormalization()(x)
        if pools != []:
            if pools[i][0]=='max':
                x = MaxPooling2D(pool_size=pools[i][1])(x)
            elif pools[i][0]=='avg':
                x = AveragePooling2D(pool_size=pools[i][1])(x)
            elif pools[i][0]=='globmax':
                x = GlobalMaxPooling2D()(x)
            elif pools[i][0]=='globavg':
                x = GlobalAveragePooling2D()(x)
        if drops != []:
            x = Dropout(drops[i])(x)

    if pools[-1][0]=='max' or pools[-1][0]=='avg':
        x = Flatten()(x)
    if end_dense != {}:
        x = Dense(end_dense['input_neurons'], activation=end_dense['activation'],name='wrap')(x)
        try:
            x = Dropout(end_dense['dropout'])(x)
        except:
            pass
    score = Dense(num_classes,activation=last_act,name='score')(x)
    
    model = Model(inpx,score)
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    return model

############################# DYNAMIC CBRNN #############################
def cbrnn_dynamic(num_classes,dimx,dimy,acts,**kwargs):
    pools     = kwargs['kwargs'].get('pools',[])
    drops     = kwargs['kwargs'].get('drops',[])
    bn        = kwargs['kwargs'].get('batch_norm',False)
    end_dense = kwargs['kwargs'].get('end_dense',{})
    last_act  = kwargs['kwargs'].get('last_act','softmax')
    
    cnn_layers = kwargs['kwargs'].get('cnn_layers',1)
    rnn_layers = kwargs['kwargs'].get('rnn_layers',1)
    rnn_type   = kwargs['kwargs'].get('rnn_type','LSTM')
    rnn_units  = kwargs['kwargs'].get('rnn_units',[])
    nb_filter     = kwargs['kwargs'].get('nb_filter',[])
    filter_length = kwargs['kwargs'].get('filter_length',[])
    #CNN with biderectional lstm
    print "CBRNN"
    if not np.all([len(acts)==cnn_layers,len(nb_filter)==cnn_layers,len(filter_length)==cnn_layers]):
        print "Layers Mismatch"
        return False
    x = Input(shape=(1,dimx,dimy),name='inpx')
    inpx = x
    for i in range(cnn_layers):
        x = Conv2D(filters=nb_filter[i],
                   kernel_size=filter_length[i],
                   data_format='channels_first',
                   padding='same',
                   activation=acts[i])(x)
        if bn:
            x=BatchNormalization()(x)
        if pools != []:
            if pools[i][0]=='max':
                x = MaxPooling2D(pool_size=pools[i][1])(x)
            elif pools[i][0]=='avg':
                x = AveragePooling2D(pool_size=pools[i][1])(x)
        if drops != []:
            x = Dropout(drops[i])(x)
    x = Permute((2,1,3))(x)
    a,b,c,d= kr(x)
    x = Reshape((b*d,c))(x)

    for i in range(rnn_layers):
        #Only last layer can have return_sequences as False
        r = False if i == rnn_layers-1 else True
        if rnn_type=='LSTM':
            x = LSTM(rnn_units[i],return_sequences=r)(x)
        elif rnn_type=='GRU':
            x = Bidirectional(GRU(rnn_units[i],return_sequences=r))(x)
        elif rnn_type=='bdLSTM':
            x = Bidirectional(LSTM(rnn_units[i],return_sequences=r))(x)
        elif rnn_type=='bdGRU':
            x = Bidirectional(GRU(rnn_units[i],return_sequences=r))(x)
    
    x= Dropout(0.1)(x)
    if end_dense != {}:
        x = Dense(end_dense['input_neurons'], activation=end_dense['activation'],name='wrap')(x)
        try:
            x = Dropout(end_dense['dropout'])(x)
        except:
            pass
    main_output = Dense(num_classes, activation=last_act, name='main_output')(x)
    model = Model(inputs=inpx, outputs=main_output)
    model.summary()
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
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
