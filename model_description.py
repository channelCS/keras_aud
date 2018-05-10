"""
Created on Sat Apr 08 11:48:18 2018

@author: Akshita Gupta
Email - akshitadvlp@gmail.com
"""

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, Conv2DTranspose, Merge
from keras.layers import BatchNormalization, ZeroPadding2D, Lambda, dot,Activation,Concatenate
from keras.layers import LSTM, GRU, Reshape, Bidirectional, Permute,TimeDistributed
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import Multiply
from keras import optimizers
from keras import backend as K
import numpy as np

############################# Keras shape #############################
def kr(t,m=None):
    if m is None:
        return t._keras_shape
    else:
        return t._keras_shape[m]

###########################FUNCTIONAL MODELS#############################################
"""
Functional Models can be accessed by 

"""
    
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
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

########################### BASIC CNN #################################

def cnn(input_neurons,dimx,dimy,dropout,nb_filter,
                         filter_length,num_classes,pool_size=(3,3),act1=None,act2=None,act3=None,dataset=None):
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
    if dataset is None:
        model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    elif dataset =='chime2016':
        model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['mse'])
    
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

############################# ADITYA: DO THIS IN DYNAMIC AND CHANGE THE CODE BACK TO CBRNN , BASIC CBRNN #############################
def cbrnn(input_neurons,dimx,dimy,num_classes,nb_filter,filter_length,act1,act2,act3,pool_size=(2,2),dropout=0.2):
    #CNN with biderectional lstm
    print "Functional CBRNN"
    main_input = Input(shape=(1,dimx,dimy))
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation='relu',use_bias=False)(main_input)
    #x1=BatchNormalization()(x)
    hx = MaxPooling2D(pool_size=pool_size)(x)
#    wrap= Dropout(dropout)(hx)
    
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation='relu',use_bias=False)(hx)
    #x2=BatchNormalization()(x)
    hx = MaxPooling2D(pool_size=pool_size)(x)
#    wrap= Dropout(dropout)(hx)
    
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation='relu',use_bias=False)(hx)
    #x3=BatchNormalization()(x)
    hx = MaxPooling2D(pool_size=(2,2))(x)
#    wrap= Dropout(dropout)(hx)
    
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation='relu',use_bias=False)(hx)
#    x4=BatchNormalization()(x)
    hx = MaxPooling2D(pool_size=(1,1))(x)
    wrap= Dropout(dropout)(x)
    
    x = Permute((2,1,3))(wrap)
    a,b,c,d= kr(x)
    x = Reshape((b*d,c))(x) 
#    x = Reshape((c*d,b))(x) 
    
    w = Bidirectional(LSTM(32,activation='sigmoid',return_sequences=False))(x)
    wrap= Dropout(dropout)(w)
    
    main_output = Dense(num_classes, activation='sigmoid', name='main_output')(wrap)
    model = Model(inputs=main_input, outputs=main_output)
    model.summary()
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['mse'])
    
    return model

####################### ATTENTION MODEL ACRNN ##################################
def block(Input):
    print kr(Input)
    cnn = Conv2D(128, (3, 3), data_format='channels_first',padding="same", activation="linear", use_bias=False)(Input)
    #cnn = BatchNormalization(axis=-1)(cnn)

    cnn1 = Lambda(slice1, output_shape=slice1_output_shape)(cnn)
    cnn2 = Lambda(slice2, output_shape=slice2_output_shape)(cnn)

    cnn1 = Activation('linear')(cnn1)
    cnn2 = Activation('sigmoid')(cnn2)

    out = Multiply()([cnn1, cnn2])
    return out

def slice1(x):
    return x[:,0:64, :, :]

def slice2(x):
    return x[:,64:128,:, :]

def slice1_output_shape(input_shape):
    return tuple([input_shape[0],64,input_shape[-2],input_shape[-1]])

def slice2_output_shape(input_shape):
    return tuple([input_shape[0],64,input_shape[-2],input_shape[-1]])

# Attention weighted sum
def outfunc(vects):
    cla, att = vects    # (N, n_time, n_out), (N, n_time, n_out)
    att = K.clip(att, 1e-7, 1.)
    out = K.sum(cla * att, axis=1) / K.sum(att, axis=1)     # (N, n_out)
    return out

def ACRNN_backup(input_neurons,dimx,dimy,num_classes,nb_filter,filter_length,act1,act2,act3,pool_size=(2,2),dropout=0.1):
    # CNN + bidirectional lstm model with attention shared across all input dimensions
    print "ACRNN"
#    main_input = Input(shape=(1,dimx,dimy))
#    x = Conv2D(filters=nb_filter,
#               kernel_size=filter_length,
#               data_format='channels_first',
#               padding='same',
#               activation='relu',use_bias=False)(main_input)
#    hx = MaxPooling2D(pool_size=pool_size)(x)
#    wrap= Dropout(dropout)(hx)
#    x = Permute((2,1,3))(wrap)
#    a,b,c,d= kr(x)
#    x = Reshape((b*d,c))(x) 
#    
#    #w = Bidirectional(LSTM(32,return_sequences=True))(x)
#    rnnout = Bidirectional(LSTM(64, activation='linear', return_sequences=True))(x)
#    rnnout_gate = Bidirectional(LSTM(64, activation='sigmoid', return_sequences=True))(x)
#    w = Multiply()([rnnout, rnnout_gate])
#    
#    
#    hidden_size = int(w._keras_shape[2])
#    hidden_states_t = Permute((2, 1), name='attention_input_t')(w)  # hidden_states_t.shape = (batch_size, hidden_size, time_steps)
#    
#    hidden_states_t = Reshape((hidden_size, hidden_states_t._keras_shape[2]), name='attention_input_reshape')(hidden_states_t)
#    # Inside dense layer
#    # a (batch_size, hidden_size, time_steps) dot W (time_steps, time_steps) => (batch_size, hidden_size, time_steps)
#    # W is the trainable weight matrix of attention
#    # Luong's multiplicative style score
#    score_first_part = Dense(hidden_states_t._keras_shape[2], use_bias=False, name='attention_score_vec')(hidden_states_t)
#    score_first_part_t = Permute((2, 1), name='attention_score_vec_t')(score_first_part)
#    #            score_first_part_t         dot        last_hidden_state     => attention_weights
#    # (batch_size, time_steps, hidden_size) dot (batch_size, hidden_size, 1) => (batch_size, time_steps, 1)
#    h_t = Lambda(lambda x: x[:, :, -1], output_shape=(hidden_size, 1), name='last_hidden_state')(hidden_states_t)
#    score = dot([score_first_part_t, h_t], [2, 1], name='attention_score')
#    attention_weights = Activation('sigmoid', name='attention_weight')(score)
#    # if SINGLE_ATTENTION_VECTOR:
#    #     a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
#    #     a = RepeatVector(hidden_size)(a)
#    # (batch_size, hidden_size, time_steps) dot (batch_size, time_steps, 1) => (batch_size, hidden_size, 1)
#    context_vector = dot([hidden_states_t, attention_weights], [2, 1], name='context_vector')
#    context_vector = Reshape((hidden_size,))(context_vector)
#    h_t = Reshape((hidden_size,))(h_t)
#    pre_activation = concatenate([context_vector, h_t], name='attention_output')
#    attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
#    main_output = Dense(num_classes, activation='softmax')(attention_vector)
#    model = Model(inputs=main_input, outputs=main_output)
#    model.summary()
#    model.compile(loss='binary_crossentropy',
#			  optimizer='adam',
#			  metrics=['mse'])
    #input_logmel = Input(shape=(dimx,dimy))
    input_logmel = Input(shape=(1,dimx,dimy))
#    a1=Permute((2,3,1))(x)

    a1 = block(input_logmel)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1,2))(a1) # (N, 240, 32, 128)
    kr(a1)
    a2 = block(a1)
    a2 = block(a2)
    a2 = MaxPooling2D(pool_size=(1,2))(a2) # (N, 240, 16, 128)
    kr(a2)    
    a3 = block(a2)
    a3 = block(a3)
    a3 = MaxPooling2D(pool_size=(1,2))(a3) # (N, 240, 8, 128)
    kr(a3)    
    a4 = block(a3)
    a4 = block(a4)
    a4 = MaxPooling2D(pool_size=(1,2))(a4) # (N, 240, 4, 128)
    kr(a4)
    a5 = Conv2D(256, (3, 3),data_format='channels_first', padding="same", activation="relu", use_bias=True)(a4)
    a5 = MaxPooling2D(pool_size=(1,2))(a5) # (N, 240, 1, 256)
    kr(a5)    
#    x=Permute((2,3,1))(a1)
    a,b,c,d=kr(a5)
    a6 = Reshape((c*d,b))(a5) # (N, 240, 256)
    kr(a6)
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(a6)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a6)
    a7 = Multiply()([rnnout, rnnout_gate])
    kr(a7)
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a7)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a7)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    kr(out)    
    model = Model([input_logmel], out)
    model.summary()
    
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model 

def ACRNN(input_neurons,dimx,dimy,num_classes,nb_filter,filter_length,act1,act2,act3,pool_size=(2,2),dropout=0.1):
    print "ACRNN"
    input_logmel = Input(shape=(1,dimx,dimy))
    a1 = block(input_logmel)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1,2))(a1) # (N, 240, 32, 128)
    kr(a1)
    a2 = block(a1)
    a2 = block(a2)
    a2 = MaxPooling2D(pool_size=(1,2))(a2) # (N, 240, 16, 128)
    kr(a2)    
    a3 = block(a2)
    a3 = block(a3)
    a3 = MaxPooling2D(pool_size=(1,2))(a3) # (N, 240, 8, 128)
    kr(a3)    
    a4 = block(a3)
    a4 = block(a4)
    a4 = MaxPooling2D(pool_size=(1,2))(a4) # (N, 240, 4, 128)
    kr(a4)
    a5 = Conv2D(256, (3, 3),data_format='channels_first', padding="same", activation="relu", use_bias=True)(a4)
    a5 = MaxPooling2D(pool_size=(1,2))(a5) # (N, 240, 1, 256)
    kr(a5)    
    a,b,c,d=kr(a5)
    a6 = Reshape((c*d,b))(a5) # (N, 240, 256)
    kr(a6)
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='relu', return_sequences=True))(a6)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a6)
    a7 = Multiply()([rnnout, rnnout_gate])
    kr(a7)
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a7)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a7)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    kr(out)    
    mymodel = Model([input_logmel], out)
    mymodel.summary()
    
    #opt=optimizers.Adam(1e-3)
    # Compile model
    mymodel.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['mse'])
    
    return mymodel 

def multi_ACRNN(input_neurons,dimx,dimy,num_classes,nb_filter,filter_length,act1,act2,act3,pool_size=(2,2),dropout=0.1):
    print "ACRNN"
    ###################################
    feat0 = Input(shape=(1,dimx,dimy[0]))
    a01 = block(feat0)
    a01 = block(a01)
    a01 = MaxPooling2D(pool_size=(1,2))(a01) # (N, 240, 32, 128)
    kr(a01)
    a02 = block(a01)
    a02 = block(a02)
    a02 = MaxPooling2D(pool_size=(1,2))(a02) # (N, 240, 16, 128)
    kr(a02)    
    a03 = block(a02)
    a03 = block(a03)
    a03 = MaxPooling2D(pool_size=(1,2))(a03) # (N, 240, 8, 128)
    kr(a03)    
    a04 = block(a03)
    a04 = block(a04)
    a04 = MaxPooling2D(pool_size=(1,2))(a04) # (N, 240, 4, 128)
    kr(a04)
    a05 = Conv2D(256, (3, 3),data_format='channels_first', padding="same", activation="relu", use_bias=True)(a04)
    a05 = MaxPooling2D(pool_size=(1,2))(a05) # (N, 240, 1, 256)
    kr(a05)    
    a,b,c,d=kr(a05)
    a06 = Reshape((c*d,b))(a05) # (N, 240, 256)
    kr(a06)
    ###################################
    feat1 = Input(shape=(1,dimx,dimy[1]))
    a1 = block(feat1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1,2))(a1) # (N, 240, 32, 128)
    kr(a1)
    a2 = block(a1)
    a2 = block(a2)
    a2 = MaxPooling2D(pool_size=(1,2))(a2) # (N, 240, 16, 128)
    kr(a2)    
    a3 = block(a2)
    a3 = block(a3)
    a3 = MaxPooling2D(pool_size=(1,2))(a3) # (N, 240, 8, 128)
    kr(a3)    
    a4 = block(a3)
    a4 = block(a4)
    a4 = MaxPooling2D(pool_size=(1,2))(a4) # (N, 240, 4, 128)
    kr(a4)
    a5 = Conv2D(256, (3, 3),data_format='channels_first', padding="same", activation="relu", use_bias=True)(a4)
    a5 = MaxPooling2D(pool_size=(1,4))(a5) # (N, 240, 1, 256)
    kr(a5)    
    a,b,c,d=kr(a5)
    a6 = Reshape((c*d,b))(a5) # (N, 240, 256)
    kr(a6)
    ##############################################
    combine = Merge(mode='concat')([a06,a6])
    ##############################################
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(combine)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(combine)
    a7 = Multiply()([rnnout, rnnout_gate])
    kr(a7)
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a7)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a7)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    kr(out)    
    mymodel = Model([feat0,feat1], out)
    mymodel.summary()
    
    opt=optimizers.Adam(1e-3)
    # Compile model
    mymodel.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    return mymodel 

############################ Transpose CNN ################################
from keras.layers import concatenate
def transpose_cnn(dimx,dimy,nb_filter,num_classes,
                         filter_length=None,pool_size=(3,3),act1=None,act2=None,act3=None,dropout=0.1):
    """
    The first section of the neural network contains conv layers.
    The deconv layer after conv layer maintains the same shape.
    The last layer will be a conv layer to calculate class wise score.
    Emphasis is given to check the size parameter for model.
    This is used for acoustic event detection.
    """
    inpx = Input(shape=(1,dimx,dimy),name='inpx')
    x = Conv2D(filters=50,
               kernel_size=5,
               data_format='channels_first',
               padding='same',
               activation='tanh')(inpx)

    hx = MaxPooling2D(pool_size=(2,1))(x)
    #hx = ZeroPadding2D(padding=(2, 1))(hx)
    hx = Conv2D(filters=100,
               kernel_size=3,
               data_format='channels_first',
               padding='same',
               activation='tanh')(hx)
   

    x=Conv2DTranspose(filters=100, kernel_size=5,padding='same', data_format='channels_first',activation='tanh')(hx)
    hx = MaxPooling2D(pool_size=(2,1))(x)
    x=Conv2DTranspose(filters=50, kernel_size=5,padding='same', data_format='channels_first',activation='tanh')(hx)
    hx = MaxPooling2D(pool_size=(2,1))(x)
    # Don't use softmax in last layer
    
    score=Conv2D(filters=num_classes, kernel_size=(1,1),padding='same', data_format='channels_first',activation='sigmoid')(hx)
    # Check for compiling    
    score=GlobalAveragePooling2D(data_format='channels_first')(score)
    kr(score)
    
    """
    x = Conv2D(filters=1,
#               kernel_size=(dimy,4),
               kernel_size=(3,1),
               data_format='channels_first',
               padding='same',
               activation='relu')(inpx)

    hx = MaxPooling2D(pool_size=(2,1))(x)
    #hx = ZeroPadding2D(padding=(2, 1))(hx)
    x = Conv2D(filters=100,
               kernel_size=(3,1),
               data_format='channels_first',
               padding='same',
               activation='sigmoid')(hx)
   
    # We apply the concept of transposed convolutional neural network for the task
    print kr(x)
    x=Conv2DTranspose(filters=100, kernel_size=(3,1),padding='same', data_format='channels_first',activation='sigmoid')(x)
    hx = MaxPooling2D(pool_size=(2,1))(x)
    x=Conv2DTranspose(filters=50, kernel_size=(3,1),padding='same', data_format='channels_first',activation='relu')(hx)
    hx = MaxPooling2D(pool_size=(2,1))(x)
    
    score=Conv2D(filters=num_classes, kernel_size=(1,1),padding='same', data_format='channels_first',activation='softmax')(hx)
    # Check for compiling    
    score=GlobalAveragePooling2D(data_format='channels_first')(score)
    kr(score)
    """    
    """
    conv_1 = Conv2D(1, (3, 3), strides=(1, 1),data_format='channels_first',  padding='same')(inpx)
    act_1 = Activation('relu')(conv_1)
    
    conv_2 = Conv2D(64, (3, 3), strides=(1, 1),data_format='channels_first',  padding='same')(act_1)
    act_2 = Activation('relu')(conv_2)

    deconv_1 = Conv2DTranspose(1, (3, 3), strides=(1, 1),data_format='channels_first',  padding='same')(act_2)
    act_3 = Activation('relu')(deconv_1)

    merge_1 = concatenate([act_3, act_1], axis=3)

    deconv_2 = Conv2DTranspose(1, (3, 3), strides=(1, 1),data_format='channels_first',  padding='same')(merge_1)
    act_4 = Activation('relu')(deconv_2)
    score=Conv2D(filters=num_classes, kernel_size=(1,1),padding='same', data_format='channels_first',activation='softmax')(act_4)
    # Check for compiling    
    score=GlobalAveragePooling2D(data_format='channels_first')(score)
    score = Dense(num_classes, activation='sigmoid')(score)
    """
    """
    h=Flatten()(hx)
    score=Dense(num_classes,activation='softmax')(h)
    """
    model = Model(inputs=[inpx], outputs=[score])
    model.summary()
    model.compile(loss='binary_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])

    return model

def transpose_cnn_back(dimx,dimy,nb_filter,num_classes,
                         filter_length=None,pool_size=(3,3),act1=None,act2=None,act3=None,dropout=0.1):
    """
    The first section of the neural network contains conv layers.
    The deconv layer after conv layer maintains the same shape.
    The last layer will be a conv layer to calculate class wise score.
    Emphasis is given to check the size parameter for model.
    This is used for acoustic event detection.
    """
    inpx = Input(shape=(1,dimx,dimy),name='inpx')
    x = Conv2D(filters=100,
               kernel_size=5,
               data_format='channels_first',
               padding='same',
               activation='tanh')(inpx)

    hx = MaxPooling2D(pool_size=(2,1))(x)
    x=Conv2DTranspose(filters=100, kernel_size=5,padding='same', data_format='channels_first',activation='tanh')(hx)
    hx = MaxPooling2D(pool_size=(2,1))(x)
    # Don't use softmax in last layer
    score=Conv2D(filters=num_classes, kernel_size=(1,1),padding='same', data_format='channels_first',activation='sigmoid')(hx)
    # Check for compiling    
    score=GlobalAveragePooling2D(data_format='channels_first')(score)
    kr(score)
    
    """
    x = Conv2D(filters=1,
#               kernel_size=(dimy,4),
               kernel_size=(3,1),
               data_format='channels_first',
               padding='same',
               activation='relu')(inpx)

    hx = MaxPooling2D(pool_size=(2,1))(x)
    #hx = ZeroPadding2D(padding=(2, 1))(hx)
    x = Conv2D(filters=100,
               kernel_size=(3,1),
               data_format='channels_first',
               padding='same',
               activation='sigmoid')(hx)
   
    # We apply the concept of transposed convolutional neural network for the task
    print kr(x)
    x=Conv2DTranspose(filters=100, kernel_size=(3,1),padding='same', data_format='channels_first',activation='sigmoid')(x)
    hx = MaxPooling2D(pool_size=(2,1))(x)
    x=Conv2DTranspose(filters=50, kernel_size=(3,1),padding='same', data_format='channels_first',activation='relu')(hx)
    hx = MaxPooling2D(pool_size=(2,1))(x)
    
    score=Conv2D(filters=num_classes, kernel_size=(1,1),padding='same', data_format='channels_first',activation='softmax')(hx)
    # Check for compiling    
    score=GlobalAveragePooling2D(data_format='channels_first')(score)
    kr(score)
    # h=Flatten()(score)
    # score=Dense(num_classes,activation='softmax')(h)
    """
    """
    conv_1 = Conv2D(1, (3, 3), strides=(1, 1),data_format='channels_first',  padding='same')(inpx)
    act_1 = Activation('relu')(conv_1)
    
    conv_2 = Conv2D(64, (3, 3), strides=(1, 1),data_format='channels_first',  padding='same')(act_1)
    act_2 = Activation('relu')(conv_2)

    deconv_1 = Conv2DTranspose(1, (3, 3), strides=(1, 1),data_format='channels_first',  padding='same')(act_2)
    act_3 = Activation('relu')(deconv_1)

    merge_1 = concatenate([act_3, act_1], axis=3)

    deconv_2 = Conv2DTranspose(1, (3, 3), strides=(1, 1),data_format='channels_first',  padding='same')(merge_1)
    act_4 = Activation('relu')(deconv_2)
    score=Conv2D(filters=num_classes, kernel_size=(1,1),padding='same', data_format='channels_first',activation='softmax')(act_4)
    # Check for compiling    
    score=GlobalAveragePooling2D(data_format='channels_first')(score)
    score = Dense(num_classes, activation='sigmoid')(score)
    """
    model = Model(inputs=[inpx], outputs=[score])
    model.summary()
    model.compile(loss='categorical_crossentropy',
			  optimizer='adadelta',
			  metrics=['accuracy'])

    return model

#####################Sequence2Sequence Model ############################

def seq2seq_lstm(input_neurons,dimx,dimy,num_classes,nb_filter,filter_length,act1,act2,act3,pool_size=(2,2),dropout=0.1):
    # Recurrent sequence to sequence learning auto encoders for audio classification task
    
    
    print "seq2seq_lstm"
    
    ## encoder
    encoder_input = Input(shape=(dimx,dimy))
    encoder=Bidirectional(LSTM(32,return_state=True),merge_mode="mul")# Returns list of nos. of output states
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_input)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]
    
#    a,b = kr(encoder_outputs)
#    x = Reshape((b,1))(encoder_outputs)
    
    
    ## decoder
    decoder_input = Input(shape=(dimx,dimy), name='main_input')
    decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_input,
                                         initial_state=encoder_states)
    #h=Flatten()(decoder_outputs)
    decoder_dense = Dense(num_classes, activation='softmax')(decoder_outputs)
    model = Model([encoder_input, decoder_input], decoder_dense)
    model.summary()
    model.compile(loss='categorical_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])

    ## encoder model 
    encoder_model = Model(encoder_input, encoder_states)
    
    ##decoder model
    
    decoder_state_input_h = Input(shape=(64,))
    decoder_state_input_c = Input(shape=(64,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_input, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model( [decoder_input] + decoder_states_inputs,[decoder_outputs] + decoder_states)
  
    return model, encoder_model, decoder_model 


########################################### DYNAMIC MODELS ###########################################
"""
Dynamic Models can be accessed by 

"""
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
    model.summary()
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
    model.summary()
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    return model

############################# DYNAMIC CBRNN #############################
def cbrnn_dynamic(num_classes,dimx,dimy,acts,**kwargs):
    """
    """
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


