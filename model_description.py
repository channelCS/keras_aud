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
def dnn(dimx,dimy,num_classes,**kwargs):
    """
    Deep Neural Network containing 3 Dense layers each followed by a Dropout.
    
    Parameters
    ----------
    input_neurons : int
        default : 200
        Number of Neurons for each Dense layer.
    dropout : float
        default : 0.1
        Dropout used after each Dense Layer.
    act1 : str
        default : relu
        Activation used after 1st layer.
    act2 : str
        default : relu
        Activation used after 2nd layer.
    act3 : str
        default : relu
        Activation used after 3rd layer.
    act4 : str
        default : softmax
        Activation used after 4th layer.
    print_sum : bool
        default : False
        Print summary if the model
    loss
        default : categorical_crossentropy
        Loss used
    optimizer
        default : adam
        Optimizer used
    metrics
        default : accuracy
        Metrics used.
        
    Returns
    -------
    DNN Model
    """
    input_neurons = kwargs['kwargs'].get('input_neurons',200)
    dropout       = kwargs['kwargs'].get('dropout',0.1)
    act1          = kwargs['kwargs'].get('act1','relu')
    act2          = kwargs['kwargs'].get('act2','relu')
    act3          = kwargs['kwargs'].get('act3','relu')
    act4          = kwargs['kwargs'].get('act4','softmax')
    print_sum      = kwargs['kwargs'].get('print_sum',False)

    loss          = kwargs['kwargs'].get('loss','categorical_crossentropy')
    optimizer     = kwargs['kwargs'].get('optimizer','adam')
    metrics       = kwargs['kwargs'].get('metrics','accuracy')
    print "Model DNN"
    print "Activation 1 {} 2 {} 3 {} 4 {}".format(act1,act2,act3,act4)
    print "Neurons {} Dropout {}".format(input_neurons,dropout)
    print "Loss {} Optimizer {} Metrics {}".format(loss,optimizer,metrics)
    input_dim = dimx * dimy
    inpx = Input(shape=(input_dim,))
    x = Dense(input_neurons, activation=act1)(inpx)
    x = Dropout(dropout)(x)
    x = Dense(input_neurons, activation=act2)(x)
    x = Dropout(dropout)(x)
    x = Dense(input_neurons, activation=act3)(x)
    x = Dropout(dropout)(x)
    score = Dense(num_classes, activation=act4)(x)
    model = Model([inpx],score)
    if print_sum:
        model.summary()
    model.compile(loss=loss,optimizer=optimizer,metrics=[metrics])
    
    return model

########################### BASIC CNN #################################
def cnn(dimx,dimy,num_classes,**kwargs):
    """
    Convolution Neural Network containing 1 Convolution layer followed by a 
    Dense Layer.
    
    Parameters
    ----------
    input_neurons : int
        default : 200
        Number of Neurons for the Dense layer.
    dropout : float
        default : 0.1
        Dropout used after the Dense Layer.
    act1 : str
        default : relu
        Activation used after 1st Convolution layer.
    act2 : str
        default : relu
        Activation used after 1st Dense layer.
    act3 : str
        default : softmax
        Activation used after last Dense layer.
    print_sum : bool
        default : False
        Print summary if the model
    nb_filter : int
        default : 100
        Number of kernels
    filter_length : int, tuple
        default : 3
        Size of kernels
    pool_size : int, tuple
        default : (2,2)
        Pooling size.
    loss
        default : categorical_crossentropy
        Loss used
    optimizer
        default : adam
        Optimizer used
    metrics
        default : accuracy
        Metrics used.
        
    Returns
    -------
    CNN Model
    """

    input_neurons  = kwargs['kwargs'].get('input_neurons',200)
    act1           = kwargs['kwargs'].get('act1','relu')
    act2           = kwargs['kwargs'].get('act2','relu')
    act3           = kwargs['kwargs'].get('act3','softmax')
    dropout        = kwargs['kwargs'].get('dropout',0.1)
    nb_filter      = kwargs['kwargs'].get('nb_filter',100)
    filter_length  = kwargs['kwargs'].get('filter_length',3)
    pool_size      = kwargs['kwargs'].get('pool_size',(2,2))
    print_sum      = kwargs['kwargs'].get('print_sum',False)

    loss          = kwargs['kwargs'].get('loss','categorical_crossentropy')
    optimizer     = kwargs['kwargs'].get('optimizer','adam')
    metrics       = kwargs['kwargs'].get('metrics','accuracy')

    print "Model CNN"
    print "Activation 1 {} 2 {} 3 {}".format(act1,act2,act3)
    print "Neurons {} Dropout {}".format(input_neurons,dropout)
    print "Kernels {} Size {} Poolsize {}".format(nb_filter,filter_length,pool_size)
    print "Loss {} Optimizer {} Metrics {}".format(loss,optimizer,metrics)
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
    if print_sum:
        model.summary()
    model.compile(loss=loss,optimizer=optimizer,metrics=[metrics])
    
    return model

########################### BASIC RNN #################################
def rnn(dimx,dimy,num_classes,**kwargs):
    """
    Deep Neural Network containing 1 LSTM layers followed by 3 Dense Layers.
    
    Parameters
    ----------
    rnn_units : int
        default : 32
        Number of Units for LSTM layer.
    input_neurons : int
        default : 200
        Number of Neurons for each Dense layer.
    act1 : str
        default : relu
        Activation used after 1st layer.
    act2 : str
        default : relu
        Activation used after 2nd layer.
    act3 : str
        default : relu
        Activation used after 3rd layer.
    act4 : str
        default : softmax
        Activation used after 4th layer.
    print_sum : bool
        default : False
        Print summary if the model
    loss
        default : categorical_crossentropy
        Loss used
    optimizer
        default : adam
        Optimizer used
    metrics
        default : accuracy
        Metrics used.
        
    Returns
    -------
    RNN Model
    """
    rnn_units     = kwargs['kwargs'].get('rnn_units',32)
    input_neurons = kwargs['kwargs'].get('input_neurons',200)
    act1          = kwargs['kwargs'].get('act1','relu')
    act2          = kwargs['kwargs'].get('act2','relu')
    act3          = kwargs['kwargs'].get('act3','relu')
    act4          = kwargs['kwargs'].get('act4','sigmoid')
    print_sum      = kwargs['kwargs'].get('print_sum',False)

    loss          = kwargs['kwargs'].get('loss','categorical_crossentropy')
    optimizer     = kwargs['kwargs'].get('optimizer','adam')
    metrics       = kwargs['kwargs'].get('metrics','accuracy')

    
    input_dim=dimx*dimy
    main_input = Input(shape=(1,input_dim), name='main_input')
    x = LSTM(rnn_units)(main_input)

    # We stack a deep densely-connected network on top
    x = Dense(input_neurons, activation=act1)(x)
    x = Dense(input_neurons, activation=act2)(x)
    x = Dense(input_neurons, activation=act3)(x)
    
    # And finally we add the main logistic regression layer
    main_output = Dense(num_classes, activation=act4, name='main_output')(x)
    model = Model(inputs=main_input, outputs=main_output)
    if print_sum:
        model.summary()
    model.compile(loss=loss,
              optimizer=optimizer,
              metrics=[metrics])

    return model

########################### BASIC CRNN #################################
def cnn_rnn(dimx,dimy,num_classes,**kwargs):
    """
    Deep Neural Network containing 1 LSTM layers followed by 3 Dense Layers.
    
    Parameters
    ----------
    rnn_units : int
        default : 32
        Number of Units for LSTM layer.
    input_neurons : int
        default : 200
        Number of Neurons for each Dense layer.
	dropout : float
        default : 0.1
        Dropout used after the Dense Layer.
    act1 : str
        default : relu
        Activation used after Convolution layer.
    act2 : str
        default : tanh
        Activation used after Recurrent layer.
    act3 : str
        default : softmax
        Activation used after Dense layer.
    print_sum : bool
        default : False
        Print summary if the model
	nb_filter : int
        default : 100
        Number of kernels
    filter_length : int, tuple
        default : 3
        Size of kernels
    pool_size : int, tuple
        default : (2,2)
        Pooling size.
    loss
        default : categorical_crossentropy
        Loss used
    optimizer
        default : adam
        Optimizer used
    metrics
        default : accuracy
        Metrics used.
        
    Returns
    -------
    RNN Model
    """
    rnn_units     = kwargs['kwargs'].get('rnn_units',32)
    input_neurons = kwargs['kwargs'].get('input_neurons',200)
    act1          = kwargs['kwargs'].get('act1','relu')
    act2          = kwargs['kwargs'].get('act2','tanh')
    act3          = kwargs['kwargs'].get('act3','softmax')
    dropout        = kwargs['kwargs'].get('dropout',0.1)
    nb_filter      = kwargs['kwargs'].get('nb_filter',100)
    filter_length  = kwargs['kwargs'].get('filter_length',3)
    pool_size      = kwargs['kwargs'].get('pool_size',(2,2))
    print_sum      = kwargs['kwargs'].get('print_sum',False)

    loss          = kwargs['kwargs'].get('loss','categorical_crossentropy')
    optimizer     = kwargs['kwargs'].get('optimizer','adam')
    metrics       = kwargs['kwargs'].get('metrics','accuracy')


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
    x = LSTM(rnn_units,activation=act2)(x)
    wrap= Dropout(dropout)(x)
    x = Dense(input_neurons, activation=act3)(wrap)
    main_output = Dense(num_classes, activation='softmax', name='main_output')(wrap)
    model = Model(inputs=main_input, outputs=main_output)
    if print_sum:
        model.summary()
    model.compile(loss=loss,
              optimizer=optimizer,
              metrics=[metrics])

    return model



############################# BASIC CBRNN #############################
def cbrnn(dimx,dimy,num_classes,**kwargs):
    """
    CNN with biderectional lstm
    
    Parameters
    ----------
    rnn_units : int
        default : 32
        Number of Units for LSTM layer.
	dropout : float
        default : 0.1
        Dropout used after the Dense Layer.
    act1 : str
        default : relu
        Activation used after 4 Convolution layers.
    act2 : str
        default : sigmoid
        Activation used after Recurrent layer.
    act3 : str
        default : sigmoid
        Activation used after Dense layer.
    print_sum : bool
        default : False
        Print summary if the model
	nb_filter : int
        default : 100
        Number of kernels
    filter_length : int, tuple
        default : 3
        Size of kernels
    pool_size : int, tuple
        default : (2,2)
        Pooling size.
    loss
        default : categorical_crossentropy
        Loss used
    optimizer
        default : adam
        Optimizer used
    metrics
        default : accuracy
        Metrics used.
        
    Returns
    -------
    CBRNN Model
    """
    rnn_units     = kwargs['kwargs'].get('rnn_units',32)
    act1          = kwargs['kwargs'].get('act1','relu')
    act2          = kwargs['kwargs'].get('act2','sigmoid')
    act3          = kwargs['kwargs'].get('act3','sigmoid')
    dropout        = kwargs['kwargs'].get('dropout',0.1)
    nb_filter      = kwargs['kwargs'].get('nb_filter',100)
    filter_length  = kwargs['kwargs'].get('filter_length',3)
    pool_size      = kwargs['kwargs'].get('pool_size',(2,2))
    print_sum      = kwargs['kwargs'].get('print_sum',False)

    loss          = kwargs['kwargs'].get('loss','binary_crossentropy')
    optimizer     = kwargs['kwargs'].get('optimizer','adam')
    metrics       = kwargs['kwargs'].get('metrics','mse')


    print "Functional CBRNN"
    main_input = Input(shape=(1,dimx,dimy))
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation=act1,use_bias=False)(main_input)
    #x1=BatchNormalization()(x)
    hx = MaxPooling2D(pool_size=pool_size)(x)
#    wrap= Dropout(dropout)(hx)
    
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation=act1,use_bias=False)(hx)
    #x2=BatchNormalization()(x)
    hx = MaxPooling2D(pool_size=pool_size)(x)
#    wrap= Dropout(dropout)(hx)
    
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation=act1,use_bias=False)(hx)
    #x3=BatchNormalization()(x)
    hx = MaxPooling2D(pool_size=(2,2))(x)
#    wrap= Dropout(dropout)(hx)
    
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation=act1,use_bias=False)(hx)
#    x4=BatchNormalization()(x)
    hx = MaxPooling2D(pool_size=(1,1))(x)
    wrap= Dropout(dropout)(x)
    
    x = Permute((2,1,3))(wrap)
    a,b,c,d= kr(x)
    x = Reshape((b*d,c))(x) 
#    x = Reshape((c*d,b))(x) 
    
#    w = Bidirectional(LSTM(rnn_units,activation=act2,return_sequences=False))(x)
    rnnout = Bidirectional(LSTM(rnn_units, activation=act2, return_sequences=True))(x)
    rnnout_gate = Bidirectional(LSTM(rnn_units, activation=act3, return_sequences=False))(x)
    w = Multiply()([rnnout, rnnout_gate])
    wrap= Dropout(dropout)(w)
    wrap=Flatten()(wrap)
    main_output = Dense(num_classes, activation=act3, name='main_output')(wrap)
    model = Model(inputs=main_input, outputs=main_output)
    model.summary()
    if print_sum:
        model.summary()
    model.compile(loss=loss,
              optimizer=optimizer,
              metrics=[metrics])
    
    return model

############################ Transpose CNN ################################
def transpose_cnn(dimx,dimy,num_classes,**kwargs):
    """
    The first section of the neural network contains conv layers.
    The deconv layer after conv layer maintains the same shape.
    The last layer will be a conv layer to calculate class wise score.
    Emphasis is given to check the size parameter for model.
    This is used for acoustic event detection.
    """
    act1          = kwargs['kwargs'].get('act1','tanh')
    act2          = kwargs['kwargs'].get('act2','tanh')
    act3          = kwargs['kwargs'].get('act3','sigmoid')
    nb_filter      = kwargs['kwargs'].get('nb_filter',[])
    pool_size      = kwargs['kwargs'].get('pool_size',(1,2))
    dropout        = kwargs['kwargs'].get('dropout',0.1)
    loss          = kwargs['kwargs'].get('loss','binary_crossentropy')
    optimizer     = kwargs['kwargs'].get('optimizer','adam')
    metrics       = kwargs['kwargs'].get('metrics','mse')
    inpx = Input(shape=(1,dimx,dimy),name='inpx')
    x = Conv2D(filters=nb_filter[0],
               kernel_size=5,
               data_format='channels_first',
               padding='same',
               activation=act1)(inpx)

    hx = MaxPooling2D(pool_size=pool_size)(x)
    #hx = ZeroPadding2D(padding=(2, 1))(hx)
    hx = Conv2D(filters=nb_filter[1],
               kernel_size=3,
               data_format='channels_first',
               padding='same',
               activation=act1)(hx)
   

    x=Conv2DTranspose(filters=nb_filter[1], kernel_size=3,padding='same', data_format='channels_first',activation=act2)(hx)
    hx = MaxPooling2D(pool_size=pool_size)(x)
    x=Conv2DTranspose(filters=nb_filter[0], kernel_size=5,padding='same', data_format='channels_first',activation=act2)(hx)
    hx = MaxPooling2D(pool_size=pool_size)(x)
    # Don't use softmax in last layer
    score=Conv2D(filters=num_classes, kernel_size=(1,1),padding='same', data_format='channels_first',activation=act3)(hx)
    # Check for compiling 
    wrap= Dropout(dropout)(score)
    score=GlobalAveragePooling2D(data_format='channels_first')(wrap)
    kr(score)
    
    model = Model(inputs=[inpx], outputs=[score])
    model.summary()
    model.compile(loss=loss,
			  optimizer=optimizer,
			  metrics=[metrics])

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

def ACRNN(dimx,dimy,num_classes,**kwargs):
    print "ACRNN"
    input_logmel = Input(shape=(1,dimx,dimy))
    a1 = block(input_logmel)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1,2))(a1)
    
    a2 = block(a1)
    a2 = block(a2)
    a2 = MaxPooling2D(pool_size=(1,2))(a2)
    
    a3 = block(a2)
    a3 = block(a3)
    a3 = MaxPooling2D(pool_size=(1,2))(a3)
    
    a4 = block(a3)
    a4 = block(a4)
    a4 = MaxPooling2D(pool_size=(1,2))(a4)
    
    a5 = Conv2D(256, (3, 3),data_format='channels_first', padding="same", activation="relu", use_bias=True)(a4)
    a5 = MaxPooling2D(pool_size=(1,2))(a5)
    
    a,b,c,d=kr(a5)
    a6 = Reshape((c*d,b))(a5)
    
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

##################### Sequence2Sequence Model ############################
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

# Beta Models

def multi_ACRNN(input_neurons,dimx,dimy,num_classes,nb_filter,filter_length,act1,act2,act3,pool_size=(2,2),dropout=0.1):
    print "ACRNN"
    ###################################
    feat0 = Input(shape=(1,dimx,dimy[0]))
    a01 = block(feat0)
    a01 = block(a01)
    a01 = MaxPooling2D(pool_size=(1,2))(a01)

    a02 = block(a01)
    a02 = block(a02)
    a02 = MaxPooling2D(pool_size=(1,2))(a02)

    a03 = block(a02)
    a03 = block(a03)
    a03 = MaxPooling2D(pool_size=(1,2))(a03)

    a04 = block(a03)
    a04 = block(a04)
    a04 = MaxPooling2D(pool_size=(1,2))(a04)

    a05 = Conv2D(256, (3, 3),data_format='channels_first', padding="same", activation="relu", use_bias=True)(a04)
    a05 = MaxPooling2D(pool_size=(1,2))(a05)

    a,b,c,d=kr(a05)
    a06 = Reshape((c*d,b))(a05)

    feat1 = Input(shape=(1,dimx,dimy[1]))
    a1 = block(feat1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1,2))(a1)
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

