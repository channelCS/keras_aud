"""
Created on Sat Apr 08 11:48:18 2018

@author: Akshita Gupta
Email - akshitadvlp@gmail.com
"""

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, Conv2DTranspose, Merge
from keras.layers import BatchNormalization, ZeroPadding2D, Lambda, dot,Activation,concatenate
from keras.layers import LSTM, GRU, Reshape, Bidirectional, Permute
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import Multiply
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

############################# BASIC CBRNN #############################
def cbrnn(input_neurons,dimx,dimy,num_classes,nb_filter,filter_length,act1,act2,act3,pool_size=(1,2),dropout=0.2):
    #CNN with biderectional lstm
    print "Functional CBRNN"
    main_input = Input(shape=(1,dimx,dimy))
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation='relu',use_bias=True)(main_input)
    #x1=BatchNormalization()(x)
    hx = MaxPooling2D(pool_size=pool_size)(x)
    wrap= Dropout(dropout)(hx)
    
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation='relu',use_bias=True)(wrap)
    #x2=BatchNormalization()(x)
    hx = MaxPooling2D(pool_size=pool_size)(x)
    wrap= Dropout(dropout)(hx)
    
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation='relu',use_bias=True)(wrap)
    #x3=BatchNormalization()(x)
    hx = MaxPooling2D(pool_size=(1,2))(x)
    wrap= Dropout(dropout)(hx)
    
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation='relu',use_bias=True)(wrap)
#    x4=BatchNormalization()(x)
    hx = MaxPooling2D(pool_size=(1,4))(x)
    wrap= Dropout(dropout)(x)
    
    x = Permute((2,1,3))(wrap)
    a,b,c,d= kr(x)
    x = Reshape((b*d,c))(x) 
    
    w = Bidirectional(LSTM(128,activation='sigmoid',return_sequences=False))(x)
    wrap= Dropout(dropout)(w)
    
    main_output = Dense(num_classes, activation='sigmoid', name='main_output')(wrap)
    model = Model(inputs=main_input, outputs=main_output)
    model.summary()
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['mse'])
    
    return model

####################### ATTENTION MODEL ACRNN ##################################

def ACRNN(input_neurons,dimx,dimy,num_classes,nb_filter,filter_length,act1,act2,act3,pool_size=(2,2),dropout=0.1):
    # CNN + bidirectional lstm model with attention shared across all input dimensions
    print "ACRNN"
    main_input = Input(shape=(1,dimx,dimy))
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation='relu')(main_input)
    hx = MaxPooling2D(pool_size=pool_size)(x)
    wrap= Dropout(dropout)(hx)
    x = Permute((2,1,3))(wrap)
    a,b,c,d= kr(x)
    x = Reshape((b*d,c))(x) 
    
    #w = Bidirectional(LSTM(32,return_sequences=True))(x)
    rnnout = Bidirectional(LSTM(128, activation='linear', return_sequences=True))(x)
    rnnout_gate = Bidirectional(LSTM(128, activation='sigmoid', return_sequences=True))(x)
    w = Multiply()([rnnout, rnnout_gate])
    
    
    hidden_size = int(w._keras_shape[2])
    hidden_states_t = Permute((2, 1), name='attention_input_t')(w)  # hidden_states_t.shape = (batch_size, hidden_size, time_steps)
    
    hidden_states_t = Reshape((hidden_size, hidden_states_t._keras_shape[2]), name='attention_input_reshape')(hidden_states_t)
    # Inside dense layer
    # a (batch_size, hidden_size, time_steps) dot W (time_steps, time_steps) => (batch_size, hidden_size, time_steps)
    # W is the trainable weight matrix of attention
    # Luong's multiplicative style score
    score_first_part = Dense(hidden_states_t._keras_shape[2], use_bias=False, name='attention_score_vec')(hidden_states_t)
    score_first_part_t = Permute((2, 1), name='attention_score_vec_t')(score_first_part)
    #            score_first_part_t         dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot (batch_size, hidden_size, 1) => (batch_size, time_steps, 1)
    h_t = Lambda(lambda x: x[:, :, -1], output_shape=(hidden_size, 1), name='last_hidden_state')(hidden_states_t)
    score = dot([score_first_part_t, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('sigmoid', name='attention_weight')(score)
    # if SINGLE_ATTENTION_VECTOR:
    #     a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    #     a = RepeatVector(hidden_size)(a)
    # (batch_size, hidden_size, time_steps) dot (batch_size, time_steps, 1) => (batch_size, hidden_size, 1)
    context_vector = dot([hidden_states_t, attention_weights], [2, 1], name='context_vector')
    context_vector = Reshape((hidden_size,))(context_vector)
    h_t = Reshape((hidden_size,))(h_t)
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    main_output = Dense(num_classes, activation='softmax')(attention_vector)
    model = Model(inputs=main_input, outputs=main_output)
    model.summary()
    model.compile(loss='binary_crossentropy',
			  optimizer='adam',
			  metrics=['mse'])
    
    return model 

def tcnn():
    print "aditya"

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

#############################  model.conv_deconv #################################




def multi_cnn(input_neurons,dimx,dimy,num_classes,nb_filter,filter_length,act1,act2,act3,pool_size=(2,2),dropout=0.1):
    # Combine different features and model according to their theoritical properties.
    # For basic, we have combined mel+ cnn & cqt +cnn in parallel
    # mini Ensemble model
    inps,outs=[],[]
    for i in range(len(dimy)):
        inpx = Input(shape=(1,dimx,dimy[i]))
        inps.append(inpx)
        x = Conv2D(filters=nb_filter,
                   kernel_size=filter_length,
                   data_format='channels_first',
                   padding='same',
                   activation=act1)(inpx)
        x = MaxPooling2D(pool_size=pool_size)(x)
        x= Dropout(dropout)(x)
        h = Flatten()(x)
        outs.append(h)

        
    
    combine = Merge(mode='concat')(outs) 
    # And finally we add the main logistic regression layer    
    wrap = Dense(input_neurons, activation=act2,name='wrap')(combine)
    main_output = Dense(num_classes,activation=act3,name='score')(wrap)
    
    model = Model(inputs=inps,outputs=main_output)
    ################################################
    model.summary()
    model.compile(loss='categorical_crossentropy',
			  optimizer='adadelta',
			  metrics=['accuracy'])
    
    
    return model

#############################  model.TCNN #################################

def sampling(args):
    epsilon_std = 1.0
    latent_dim = 2
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

def transpose_cnn(dimx,dimy,nb_filter,filter_length,num_classes,pool_size=(3,3),dropout=0.1):
    """
    First section contains CNN.
    Deconv layer will be after conv layer to maintain the same shape.
    The last layer will be a conv layer to calculate class wise score
    """
    nb_filter = 64
    num_conv = 3
    intermediate_dim = 128
    latent_dim = 2
    batch_size = 100
    inpx = Input(shape=(1,dimx,dimy),name='inpx')
    
    conv_1 = Conv2D(filters=1,
               kernel_size=(2,2),
               data_format='channels_first',
               padding='same',
               activation='relu')(inpx)

    conv_2 = Conv2D(filters=nb_filter,
               kernel_size=(2,2),
               data_format='channels_first',
               padding='same',
               activation='relu')(conv_1)

    conv_3 = Conv2D(filters=nb_filter,
               kernel_size=num_conv,
               data_format='channels_first',
               padding='same',
               activation='relu')(conv_2)

    conv_4 = Conv2D(filters=nb_filter,
               kernel_size=num_conv,
               data_format='channels_first',
               padding='same',
               activation='relu')(conv_3)
    
    flat = Flatten()(conv_4)
    hidden = Dense(intermediate_dim, activation='relu')(flat)

    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_hid = Dense(intermediate_dim, activation='relu')
    decoder_upsample = Dense(nb_filter * dimx * dimy, activation='relu')

    output_shape = (batch_size, nb_filter, dimx, dimy)
    
    decoder_reshape = Reshape(output_shape[1:])
    decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
    decoder_deconv_2 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
    output_shape = (batch_size, filters, 29, 29)
# In this section we apply the concept of transposed convolutional neural network for the task
# of event detection, deconv will work with tensorflow 
    print kr(x)
    x=Conv2DTranspose(filters=nb_filter,padding='same', data_format='channels_first',activation='relu')(x)
    hx = MaxPooling2D(pool_size=(4,1))(x)
    x=Conv2DTranspose(filters=nb_filter, kernel_size=256,strides=(4,1),padding='same', data_format='channels_first',activation='relu')(hx)
    hx = MaxPooling2D(pool_size=(4,1))(x)
    
    score=Conv2D(filters=num_classes, kernel_size=256,padding='same', data_format='channels_first',activation='sigmoid')(hx)
# Check for compiling    
    model = Model([inpx],score)
    model.compile(loss='categorical_crossentropy',
			  optimizer='adadelta',
			  metrics=['accuracy'])
  
    return model
