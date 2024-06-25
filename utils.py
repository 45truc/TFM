from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.signal as spsg

def plot_(array, channel, trial, block):
    y = array[channel, :, trial, block]
    x = range(len(y))
    plt.figure(figsize=(14, 4)) 
    plt.plot(x,y, label=f"Channel: {channel}; Trial: {trial}; Block : {block}")

    plt.xlabel('t')
    plt.ylabel('Signal')
    plt.title('Time Series Plot')
    
    # Adding grid
    plt.grid(True)
    plt.legend()
    return 

def summary(array, text=True):
    shape = array.shape
    C = shape[0] #Channels (signal), 
    T = shape[1] #time,
    Trials = shape[2] # trial,
    B = shape[3] #trialblock
    empty = np.zeros((B))
    dict = {}
    for b in range(B):
        channels = []
        for c in range(C):
            if np.all(np.isnan(array[c,:,0,b])):
               channels.append(c)
               empty[b] = len(channels)
               dict.update({f"{b}": channels})
    #print(f"Total empty channels: {np.sum(empty)*}")
    #print(f"Average empty channels: {np.sum(empty)/(b*t)}")
    if text:
        print(f"Channels: {C}, Trials: {Trials}, TrialBlocks: {B}")
        print(f"Number of active channels for each trial block: {60-empty}")
    return dict

def get_data_subject(data, sub, selection_critirium=None, channels_only=False):
    
    channel_info = summary(data['dataSorted'], text=False) 
    array_tuples = [tuple(array) for array in channel_info.values()]
    unique_tuples = set(array_tuples)
    unique_arrays = [np.array(tuple_) for tuple_ in unique_tuples]

    #Dealing with problematic subjects
    blocks = np.array(range(12))
    #From the data exploraion
    problematic = {1:{4,9}, 4:{2,9}, 6:{5,11}}
    if sub in {1,4,6}:
        blocks = np.array([x for x in blocks if not(x in problematic[sub])])
        #Eliminate arrays with all channels    
        unique_arrays = [x for x in unique_arrays if len(x)<59]
        
    if selection_critirium==None:
        #Select the set of missing chanels
        #In this case we take all blocks unless, probelamtics need to be seen separetly
        seto = set()
        for arr in unique_arrays:
            seto = seto | set(arr)
        select = np.array(list(seto))
    else: 
        #In this case we follow least number of missing channels
        select = min(unique_arrays, key=len)
        # blocks is the blocks we will use
        blocks = [key for key, value in channel_info.items() if set(value) == set(select)]
        blocks = np.array(blocks, int)
        
    active_channels = [x for x in range(60) if x not in select]

    if channels_only:
        return active_channels
        
    return data['dataSorted'][active_channels][...,blocks], blocks

def Generate_data(blocks):
    labels = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
    filtered_labels = []
    for b in blocks:
        filtered_labels.append(labels[b])
    
    #re_sample and blocks already loaded
    Y_sample = np.repeat(filtered_labels, 108)
    Y_hot = torch.Tensor(np.eye(3)[Y_sample])
    return Y_sample, Y_hot 

def Load_Data(subj=0, data_path = f"E:\data/", Ffilter=False):
    '''
     data_path = f"E:\data/dataClean-ICA3-{num}-T1.mat"
     subj = 0,...,10
     num = 25,...,35
    '''
    data = loadmat(data_path+f'dataClean-ICA3-{25+subj}-T1.mat')
    sample, blocks = get_data_subject(data, sub=subj, selection_critirium=None)
    # Some nans may still remain
    #if np.any(np.isnan(sample)):
    #    sample = np.nan_to_num(sample, nan=0.0)
        
    sha = sample.shape
    sample = np.reshape(sample, (sha[0],sha[1],sha[2]*sha[3]), order='F').T
    sample = np.moveaxis(sample, 1, 2)
    sample = torch.Tensor(sample)
    
    #Filtering out the remaining nans due to errors in specific trials
    without_nans = [index for index in range(sample.shape[0]) if not torch.any(torch.isnan(sample[index]))]

    Y_sample, Y_hot = Generate_data(blocks)

    if Ffilter:
        return filter_data(sample[without_nans]), Y_sample[without_nans], Y_hot[without_nans]
    else:
        return sample[without_nans], Y_sample[without_nans], Y_hot[without_nans]
        

def bandpass_filtering(lfp, frq=1000, lp=250., hp=500.):
    '''
    Manuel
    '''
    b,a = spsg.iirfilter(3, [lp/frq,hp/frq], btype='bandpass', ftype='butter')
    filtered = np.empty(lfp.shape)
    for i, d in enumerate(lfp):
        # print(i, end=" ")
        filtered[i] = spsg.filtfilt(b, a, d)
    return filtered

def filter_data(re_sample):
    filtered = np.zeros(re_sample.shape)
    for s in range(re_sample.shape[0]):
        filtered[s] = bandpass_filtering(re_sample[s], frq=1000, lp=250., hp=500.)
    return filtered