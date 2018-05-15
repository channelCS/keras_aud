# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 15:26:13 2018

@author: @author: Aditya Arora
Email - adityadvlp@gmail.com
"""
from keras import backend as K
import numpy as np
from sklearn.metrics import roc_curve
import modules as M
import csv
from glob import glob
from shutil import copyfile

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
    if feature in M.get_list():
        yaml_load=M.read_yaml(yaml_file)
    n1 = yaml_load[feature][find][0]
    if n1 != dimy:
        print "Dimension Mismatch. Expected {} Found {}".format(n1,dimy)
        raise SystemExit
    else:
        print "Correct dimension"

def calculate_accuracy(truth,pred):   
    """   
    Input:
    Output:
        
    """
    pos,neg=0,0 
    for i in range(0,len(pred)):
        if pred[i] == truth[i]:
            pos = pos+1
        else:
            neg = neg+1
    acc=(float(pos)/float(len(pred)))*100
    return acc

def prec_recall_fvalue(pred, truth, thres, average):
    """
    Args:
      pred: shape = (n_samples,) or (n_samples, n_classes)
      truth: shape = (n_samples,) or (n_samples, n_classes)
      thres: float between 0 and 1. 
      average: None (element wise) | 'micro' (calculate metrics globally) 
        | 'macro' (calculate metrics for each label then average). 
      
    Returns:
      prec, recall, fvalue | list or prec, recall, fvalue. 
    """
    eps = 1e-10
    if pred.ndim == 1:
        (tp, fn, fp, tn) = cal_tp_fn_fp_tn(pred, truth, thres, average=None)
        prec = tp / max(float(tp + fp), eps)
        recall = tp / max(float(tp + fn), eps)
        fvalue = 2 * (prec * recall) / max(float(prec + recall), eps)
        return prec, recall, fvalue
    elif pred.ndim == 2:
        n_classes = pred.shape[1]
        if average is None or average == 'macro':
            precs, recalls, fvalues = [], [], []
            for j1 in xrange(n_classes):
                (prec, recall, fvalue) = prec_recall_fvalue(pred[:, j1], truth[:, j1], thres, average=None)
                precs.append(prec)
                recalls.append(recall)
                fvalues.append(fvalue)
            if average is None:
                return precs, recalls, fvalues
            elif average == 'macro':
                return np.mean(precs), np.mean(recalls), np.mean(fvalues)
        elif average == 'micro':
            (prec, recall, fvalue) = prec_recall_fvalue(pred.flatten(), truth.flatten(), thres, average=None)
            return prec, recall, fvalue
        else:
            raise Exception("Incorrect average arg!")
    else:
        raise Exception("Incorrect dimension!")        
        
def cal_tp_fn_fp_tn(pred, truth, thres, average):
    """
    Args:
      pred: shape = (n_samples,) or (n_samples, n_classes)
      truth: shape = (n_samples,) or (n_samples, n_classes)
      thres: float between 0 and 1. 
      average: None (element wise) | 'micro' (calculate metrics globally) 
        | 'macro' (calculate metrics for each label then average). 
      
    Returns:
      tp, fn, fp, tn or list of tp, fn, fp, tn. 
    """
    if pred.ndim == 1:
        y_pred = np.zeros_like(pred)
        y_pred[np.where(pred > thres)] = 1.
        tp = np.sum(y_pred + truth > 1.5)
        fn = np.sum(truth - y_pred > 0.5)
        fp = np.sum(y_pred - truth > 0.5)
        tn = np.sum(y_pred + truth < 0.5)
        return tp, fn, fp, tn
    elif pred.ndim == 2:
        tps, fns, fps, tns = [], [], [], []
        n_classes = pred.shape[1]
        for j1 in xrange(n_classes):
            (tp, fn, fp, tn) = cal_tp_fn_fp_tn(pred[:, j1], truth[:, j1], thres, None)
            tps.append(tp)
            fns.append(fn)
            fps.append(fp)
            tns.append(tn)
        if average is None:
            return tps, fns, fps, tns
        elif average == 'micro' or average == 'macro':
            return np.sum(tps), np.sum(fns), np.sum(fps), np.sum(tns)
        else: 
            raise Exception("Incorrect average argument!")
    else:
        raise Exception("Incorrect dimension!")    
    
def calculate_eer(te_y,y_pred):
    """   
    Input:
    Output:
        
    """
    x = len(te_y[0]) #num classes
    eps = 1E-6
    class_eer=[]
    for k in xrange(x):
        f, t, _ = roc_curve(te_y[:,k], y_pred[:,k]) #it takes 1d array as input
        Points = [(0,0)]+zip(f,t)
        for i, point in enumerate(Points):
            if point[0]+eps >= 1-point[1]:
                break
        P1 = Points[i-1]; P2 = Points[i]
            
        #Interpolate between P1 and P2
        if abs(P2[0]-P1[0]) < eps:
            ER = P1[0]        
        else:        
            m = (P2[1]-P1[1]) / (P2[0]-P1[0])
            o = P1[1] - m * P1[0]
            ER = (1-o) / (1+m) 
        class_eer.append(ER)
    
    EER=np.mean(class_eer)
    return EER

def get_activations(model, layer, X_batch):
    """   
    Input:
    Output:
        
    """
    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch,0])
    print(activations)
    return activations

def mat_2d_to_3d(X, agg_num, hop):
    """   
    Input:
    Output:
        
    """
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
    """   
    Input:
    Output:
        
    """
    [batch_num, dimx, dimy]= X.shape 
    two_d   = ['DNN']
    three_d = ['RNN']
    four_d  = ['CNN', 'CHOU', 'CRNN', 'FCRNN', 'CBRNN', 'MultiCNN', 'TCNN','ACRNN', 'MultiACRNN']
    if model in two_d:
        X = X.reshape(batch_num, dimx*dimy)    
    elif model in three_d:
        X = X.reshape((batch_num,1,dimx*dimy))
    elif model in four_d:
        X = X.reshape((batch_num,1,dimx,dimy))
        
    return X

def equalise(tr_X):
    """   
    Input:
    Output:
        
    """
    chan=[]
    #l=len(max(tr_X[:]))
    m=-1
    l=-1
    for i in range(len(tr_X)):
        if len(tr_X[i])>m:
            m=len(tr_X[i])
            l=i

    chan=[i for i in range(len(tr_X)) if len(tr_X[i])!=m]
    for k in chan:
        a,b,d=tr_X[k].shape
        newx=np.zeros([m,b,d])
        j=0
        for i in range(len(tr_X[k])):
            newx[j]=tr_X[k][i]
            newx[j+1]=tr_X[k][i]
            j+=2
        tr_X[k]=newx
        
    return tr_X

def unpack_chime_2k16(path,wav_dev_fd,wav_eva_fd,meta_train_csv,meta_test_csv,label_csv):
    """   
    Input:
    Output:
        
    """
    p=path+'/chime_home'
    folder1='/'.join(meta_train_csv.split('/')[:-1])
    M.CreateFolder(folder1)
    M.CreateFolder(wav_eva_fd)
    M.CreateFolder(wav_dev_fd)
    M.CreateFolder(label_csv)
    copyfile(p+'/development_chunks_refined.csv',meta_train_csv)
    copyfile(p+'/evaluation_chunks_refined.csv',meta_test_csv)

    old_path1=path+'/chime_home/chunks'
    old_path_16=old_path1+'/*.16KHz.wav'
    old_path_48=old_path1+'/*.48KHz.wav'
    old_path_csv=old_path1+'/*.csv'
    i=0
    for f in glob(old_path_16):
        i+=1
    print "Files at 16KHz: ",i
    i=0
    for f in glob(old_path_48):
        i+=1
    print "Files at 48KHz: ",i
    with open( meta_test_csv, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    for li in lis:
        old_path=old_path1+'/'+li[1]+'.16KHz.wav'
        new_path=wav_eva_fd+'/'+li[1]+'.wav'
        copyfile(old_path,new_path)    

    with open( meta_train_csv, 'rb') as g:
        reader2 = csv.reader(g)
        lis2 = list(reader2)
    for li in lis2:
        old_path=old_path1+'/'+li[1]+'.48KHz.wav'
        new_path=wav_dev_fd+'/'+li[1]+'.wav'
        copyfile(old_path,new_path) 
        
    for f in glob(old_path_csv):
        g = label_csv+'/'+ f.split('\\')[-1]
        copyfile(f,g) 
