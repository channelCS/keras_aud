'''
SUMMARY:  importing file
AUTHOR:   adityac8
Created:  2018.03.28
Modified: 2018.03.28
--------------------------------------
'''

from __future__ import print_function
from __future__ import division
import os
import pickle
import modules as M
import wavio 

def get_samp(path):
    """   
    Input: str
        path to audio file
    Output: int
        sampling rate of audio
    """
    try:
        Struct = wavio.read( path )
        fs = Struct.rate
    except:
        raise Exception("File not found",path)
    return fs

def call_ftr(feature_name,featx,wav_fd,fe_fd,library,print_arr,dataset):
    flag1 = True if 'names' in print_arr else False
    flag2 = True if 'shape' in print_arr else False
    try:
        M.CreateFolder(fe_fd)
        M.rem_all_files(fe_fd)
    except Exception as e:
        raise Exception("Error.",e)
    
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.wav') ]
    names = sorted(names)
    if names==[]:
        raise Exception('Empty folder found!! Try Again.')
    for na in names:
        path = wav_fd + '/' + na
        # Introduce features here
        X=M.call_ftr_one(feature_name,featx,path,library,dataset)
        if flag1:
            print(na)
        if flag2:
            print(X.shape)
        out_path = fe_fd + '/' + na[0:-4] + '.f'
        pickle.dump( X, open(out_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL )
    print("extraction complete!")

def extract(feature_name,wav_fd=None,fe_fd=None,yaml_file='',library='wavread',print_arr=[],dataset=None):
    """
    This function extracts features from audio.

    Parameters
    ----------
        feature_name : str
            Name of feature to be extracted
        wav_fd : str
            Path to audio files
        fe_fd : str
            Path to feature files
        yaml_file : str
            Path to yaml file
        print_arr : str array, optional
            Description of feature that should be printed 
    """
    # Introduce features here
    if dataset is not None:
        print("Dataset called is",dataset)
    if feature_name in M.get_list():
        yaml_load=M.read_yaml(yaml_file)
        try:            
            featx=yaml_load[feature_name]
        except Exception as e:
            print("Make sure you add the {} to the YAML file".format(e))
            raise SystemExit
        x=call_ftr(feature_name,featx,wav_fd,fe_fd,library,print_arr,dataset)
        print("Something wrong happened" if x==1000 else "Feature found")
    else:
        print("Invalid Feature Name")
