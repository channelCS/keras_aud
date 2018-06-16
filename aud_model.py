"""
Created on Sat Apr 08 11:48:18 2018
author: @akshitac8
"""

import model_description as M
from keras import backend as K
K.set_image_dim_ordering('th')
   
class Functional_Model:
    """
    Class for functional model.
    
    Supported Models
    ----------
    DNN      : Deep Neural Network

    CNN      : Convolution Neural Network
    
    RNN      : Recurrent Neural Network
    
    CRNN     : Convolution Recurrent Neural Network
    
    CBRNN    : Bi-Directional Convolution Recurrent Neural Network
    
    MultiCNN : Multi Feature Convolution Neural Network
    
    ACRNN    : Attention Based Convolution Recurrent Neural Network
    
    TCNN     : Transpose Convolution Neural Network


    
    Parameters
    ----------
    model : str
        Name of Model
    dimx : int
        Second Last Dimension
    dimy : int
        Last Dimension
    num_classes : int
        Number of Classes
        
    Returns
    -------
    Functional Model
    """
    def __init__(self,model,dimx,dimy,num_classes,**kwargs):
        if model is None:
            raise ValueError("No model passed")
        self.model=model
        self.dimx = dimx
        self.dimy = dimy
        self.num_classes=num_classes
        self.kwargs=kwargs

    def prepare_model(self):
        """
        This function
        """
        if self.model=='DNN':
            lrmodel=M.dnn(num_classes = self.num_classes, dimx = self.dimx, dimy = self.dimy, kwargs = self.kwargs)
        elif self.model=='CNN':
            lrmodel=M.cnn(num_classes = self.num_classes, dimx = self.dimx, dimy = self.dimy, kwargs = self.kwargs)
        elif self.model=='RNN':
            lrmodel=M.rnn(num_classes = self.num_classes, dimx = self.dimx, dimy = self.dimy, kwargs = self.kwargs)
        elif self.model=='CRNN':
            lrmodel=M.cnn_rnn(num_classes = self.num_classes, dimx = self.dimx, dimy = self.dimy, kwargs=self.kwargs)
        elif self.model=='FCRNN':
            lrmodel=M.feature_cnn_rnn(num_classes = self.num_classes, dimx = self.dimx,dimy = self.dimy, kwargs=self.kwargs)
        elif self.model=='CBRNN':
            lrmodel=M.cbrnn(num_classes = self.num_classes, dimx = self.dimx,dimy = self.dimy, kwargs=self.kwargs)
        elif self.model=='ParallelCNN':
            lrmodel=M.parallel_cnn(num_classes = self.num_classes, dimx = self.dimx, dimy = self.dimy, kwargs=self.kwargs)
        elif self.model=='MultiCNN':
            lrmodel=M.multi_cnn(num_classes = self.num_classes, dimx = self.dimx, dimy = self.dimy, kwargs=self.kwargs)
        elif self.model=='ACRNN':
            lrmodel=M.ACRNN(num_classes = self.num_classes, dimx = self.dimx, dimy = self.dimy, kwargs=self.kwargs)
        elif self.model=='TCNN':
            lrmodel=M.transpose_cnn(num_classes = self.num_classes, dimx = self.dimx, dimy = self.dimy, kwargs=self.kwargs)
        elif self.model=='MultiACRNN':
            lrmodel=M.multi_ACRNN(num_classes = self.num_classes, dimx = self.dimx, dimy = self.dimy, kwargs=self.kwargs)
        elif self.model=='seq2seq':
            lrmodel=M.seq2seq(num_classes = self.num_classes, dimx = self.dimx, dimy = self.dimy, kwargs=self.kwargs)
        
        else:
            raise ValueError("Could not find model {}".format(self.model))
        return lrmodel
         
                
class Dynamic_Model:
    def __init__(self,model,num_classes,dimx,dimy,acts,**kwargs):
        if model is None:
            raise ValueError("No model passed")
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
            else:
                raise ValueError("Could not find model {}".format(self.model))
            return lrmodel
        except Exception as e:
            print(e)
          
