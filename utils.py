from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim

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

def get_data_subject(data, sub, selection_critirium=None):
    
    def Check_Problematic(sub,block):
        if not(sub in range(0,11)):
            sub = sub - 25
        return    (sub==1 and (block==4 or block ==9))or(
                   sub==4 and (block==2 or block ==9)
                  )or(sub==6 and (block==5 or block ==11))
    
    channel_info = summary(data['dataSorted'], text=False) 
    array_tuples = [tuple(array) for array in channel_info.values()]
    unique_tuples = set(array_tuples)
    unique_arrays = [np.array(tuple_) for tuple_ in unique_tuples]
    
    if selection_critirium==None:
        #Select the set of missing chanels
        #In this case we take all blocks unless, probelamtis need to be seen separetly
        seto = set()
        for arr in unique_arrays:
            seto = seto | set(arr)
        select = np.array(list(seto))
        blocks = np.array(range(12))
    else: 
        #In this case we follow least number of missing channels
        select = min(unique_arrays, key=len)
        # blocks is the blocks we will use
        blocks = [key for key, value in channel_info.items() if set(value) == set(select)]
        blocks = np.array(blocks, int)
        
    active_channels = [x for x in range(60) if x not in select]
    #if Check_Problematic(sub,block)
    
    return data['dataSorted'][active_channels][...,blocks], blocks