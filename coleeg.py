__version__ = 2.0

import mne
import numpy as np
from datetime import datetime, timedelta
import scipy.io
import sys
import time
import os
from IPython import get_ipython
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Lambda
from keras import activations
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv3D, InputLayer, Conv1D
from tensorflow.keras.layers import MaxPooling3D, AveragePooling3D, AveragePooling1D
from tensorflow.keras.layers import Conv2D, LSTM
from tensorflow.keras.layers import Reshape, GlobalAveragePooling1D, TimeDistributed, Conv2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import gc
import matplotlib.pyplot as plt
import csv

############################ Function Start #################################
def coleeg_info():
  info = """
* Coleeg is an open source initiative for collaborating on making a piece of software for investigating the use of Neural Networks in EEG signal classification on different datasets.

* License GPL V2.0

## Team:
Mahmoud Alnaanah (malnaanah@gmail.com)
Moutz Wahdow (m.wahdow@gmail.com)

## Team:

Mahmoud Alnaanah (malnaanah@gmail.com)
Moutz Wahdow (m.wahdow@gmail.com)

## How to install Coleeg to your google drive

  1- Make a new directory called "coleeg" on your google drive
  2- Copy Coleeg version folder, i.e. "ver2", inside the folder "coleeg'

## How to install dataset files on your Google drive

  * Physionet
    1- Go to the link: https://physionet.org/static/published-projects/eegmmidb/eeg-motor-movementimagery-dataset-1.0.0.zip
    2- Upload dataset file to your google drive inside the folder "datasets"

  * BCI competetion IV-2a  
    1- Got to the link: http://www.bbci.de/competition/iv/#download
    2- Enter your name and email address and press "I Agree"
    3- Check your email inbox and spam folder, you will get a link with the username and the password to download the dataset file.
    4- Download the dataset file.
    5- Download the labels file from the link: http://www.bbci.de/competition/iv/results/#labels
    6- Upload dataset and labels files to your google drive inside the folder "datasets"
  
    * TTK dataset is a private dataset and (up to now) not available publicly.

## What is new ne Version 2 of Coleeg
  * Two models were added, which are: CNN1D_MFBF and EEGNet_fusion.
  * A new dataset (TTK) from the Hungarian Academy of Sciences Research Centre for
    Natural Sciences (MTA-TTK) is added to the datasets.
  * New utility functions were also added, which are:
    1- divide_time: used to divide the time of a trail event into a multiple portions and that could
      be used for data augmentation.
    2- evaluate_model: model evaluation is now carried out using this function instead of doing it
      in the application file (as in the first release).
    3- get_data_ttk: used to load the data from the MTA-TTK dataset.
    4- plot_results: used to plot the evaluation accuracy vs training epochs.
    5- save_results: used to save the evaluation results to Google Drive.

## Links:
Human activity recognition (similar problem to EEG classification)
* https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/

Example on using CNN1D for reading thoughts 
* https://medium.com/@justlv/using-ai-to-read-your-thoughts-with-keras-and-an-eeg-sensor-167ace32e84a

Video classification approaches
* http://francescopochetti.com/video-classification-in-keras-a-couple-of-approaches/

MNE related links
* https://mne.tools/stable/auto_examples/time_frequency/plot_time_frequency_global_field_power.html?highlight=alpha%20beta%20gamma

* https://mne.tools/stable/auto_tutorials/preprocessing/plot_30_filtering_resampling.html#sphx-glr-auto-tutorials-preprocessing-plot-30-filtering-resampling-py

K-Fold cross-validation
* https://androidkt.com/k-fold-cross-validation-with-tensorflow-keras/

Fix memory leak in Colab
* https://github.com/tensorflow/tensorflow/issues/31312
* https://stackoverflow.com/questions/58137677/keras-model-training-memory-leak

Difference between Conv2D DepthWiseConv2D and SeparableConv2D
* https://amp.reddit.com/r/MLQuestions/comments/gp2pj9/what_is_depthwiseconv2d_and_separableconv2d_in/

Model difinition for EEGNet, ShallowConvNet, DeepConvNet was taken from the following link:
* https://github.com/rootskar/EEGMotorImagery

  """
  print(info)
  return
############################ Function End #################################


############################ Function Start #################################
def get_data_ttk(Subjects = np.arange(1,26), Exclude=None, data_path='/gdrive/MyDrive/datasets/TTK/',
                        Bands=None, resample_freq=None, Data_type=np.float32, tmin=0, tmax=2,Baseline=None, notch_freqs=None):
  # First subject in Subjects and Exclude has number = 1

  if Exclude is not None:
    print('Excluded Subjects are '+ str(Exclude))
    Subjects = np.delete(Subjects,np.isin(Subjects,Exclude))
  else:
    print('No subject excluded')

  subject_count = 0
  data_index = np.zeros((len(Subjects),2)).astype(np.int) # starting data index for each subject and its number
  data_index[:,1] = Subjects

  data_count = 0
  for sub_index, subject in enumerate(Subjects):
    print(f'\rLoading subject {sub_index+1}/{len(Subjects)}', end = '')
    data_index[subject_count,0] = data_count
    subject_count+=1   
    if subject == 1:
      raw_fnames = [f'{data_path}subject{subject:02d}/rec01.vhdr', f'{data_path}subject{subject:02d}/rec02.vhdr', f'{data_path}subject{subject:02d}/rec03.vhdr']
      raw = mne.io.concatenate_raws([mne.io.read_raw_brainvision(f, preload=True, verbose=False) for f in raw_fnames])
    elif subject == 9:
      # rec02.vhdr is skipped because it is has a problem
      raw_fnames = [f'{data_path}subject{subject:02d}/rec01.vhdr']
      raw = mne.io.concatenate_raws([mne.io.read_raw_brainvision(f, preload=True, verbose=False) for f in raw_fnames])
    elif subject == 10:
      file_name = f'{data_path}subject{subject:02d}/rec01-uj.vhdr'
      raw = mne.io.read_raw_brainvision(file_name, preload=True, verbose=False)
    elif subject == 17:
      raw_fnames = [f'{data_path}subject{subject:02d}/rec01.vhdr', f'{data_path}subject{subject:02d}/rec02.vhdr']
      raw = mne.io.concatenate_raws([mne.io.read_raw_brainvision(f, preload=True, verbose=False) for f in raw_fnames])
    else:
      file_name = f'{data_path}subject{subject:02d}/rec01.vhdr'
      raw = mne.io.read_raw_brainvision(file_name, preload=True, verbose=False)

    raw.load_data() # needed for filteration
    if resample_freq is not None:
      raw.resample(resample_freq, npad="auto")
    if notch_freqs != None:
        raw.notch_filter(notch_freqs)
    event_dict = {'Stimulus/S  1':1, 'Stimulus/S  5':2, 'Stimulus/S  7':3, 'Stimulus/S  9':4, 'Stimulus/S 11':5} 
    events, _ = mne.events_from_annotations(raw,event_id = event_dict, verbose=False)
    if Bands is not None:
      # Getting epochs for different frequncy bands   
      for band, (fmin, fmax) in enumerate(Bands):
        raw_bandpass = raw.copy().filter(fmin, fmax, fir_design='firwin')
        epochs = mne.Epochs(raw_bandpass, events, event_id=event_dict,baseline=Baseline, tmin=tmin, tmax=tmax, preload=True, event_repeated = 'drop',verbose=False)
        if not 'bands_data' in locals():
          D_SZ = epochs.get_data().shape
          bands_data = np.empty((D_SZ[0],D_SZ[2],D_SZ[1],len(Bands)))
        # Swapping dimensions from (epoch, channel, sample) to (epoch, sample, channel)
        bands_data[:,:,:,band] = epochs.get_data().transpose(0,2,1) 
        del raw_bandpass
    else:
      epochs = mne.Epochs(raw, events, event_id=event_dict,baseline=Baseline, tmin=tmin, tmax=tmax, preload=True, event_repeated = 'drop',verbose=False)
      bands_data = epochs.get_data()
      SZ = bands_data.shape
      # Swapping dimensions from (epoch, channel, sample) to (epoch, sample, channel)
      bands_data = bands_data.transpose(0,2,1).reshape(SZ[0],SZ[2],SZ[1],1)
    

    tmp_y = epochs.events[:,2]

    # Adjusting events numbers to be compatible with output classes numbers
    tmp_y = tmp_y - 1

    max_epochs = (284+65*4)
    SZ = bands_data.shape      
    # Creating output x and y matrices
    if not 'data_x' in locals():
      data_x = np.empty((max_epochs*len(Subjects),SZ[1],SZ[2],SZ[3]),dtype=Data_type)
      data_y = np.empty(max_epochs*len(Subjects),dtype=np.uint16)

    ## adjusting data type
    bands_data = bands_data.astype(Data_type)
    tmp_y = tmp_y.astype(np.uint16)

  
    data_x[data_count:data_count + SZ[0],:,:,:] = bands_data
    data_y[data_count:data_count + SZ[0]] = tmp_y

    data_count+=SZ[0]

    del raw, epochs, tmp_y, bands_data # saving memory 


  data_x = data_x[0:data_count,:,:,:]
  data_y = data_y[0:data_count]

  return data_x, data_y, data_index

########################### Function End ####################################



############################ Function Start #################################
def get_data_bcicomptIV2a(Subjects = np.arange(1,19), Exclude=None, data_path='/content/bcicomptIV2a/',
                        Bands=None, resample_freq=None, Data_type=np.float32, tmin=0, tmax=4,Baseline=None, notch_freqs=None):

  # First subject in Subjects and Exclude has number = 1

  if Exclude is not None:
    print('Excluded Subjects are '+ str(Exclude))
    Subjects = np.delete(Subjects,np.isin(Subjects,Exclude))
  else:
    print('No subject excluded')

  subject_count = 0
  data_index = np.zeros((len(Subjects),2)).astype(np.int) # starting data index for each subject and its number
  data_index[:,1] = Subjects
  data_count = 0
  for sub_index,subject in enumerate(Subjects):
    print(f'\rLoading subject {sub_index+1}/{len(Subjects)}', end = '')
    data_index[subject_count,0] = data_count
    subject_count+=1

    if subject in range(1,10):
      dataset_type = 'T'
      sub =subject
      event_dict = {'769':1, '770':2, '771':3, '772':4}
    else:
      dataset_type = 'E'
      sub =subject - 9
      event_dict= {'783':1}

    file_name = f'{data_path}A{sub:02d}{dataset_type}.gdf'
    raw = mne.io.read_raw_gdf(file_name,verbose='ERROR')
    raw.load_data() # needed for filteration
    if resample_freq is not None:
      raw.resample(resample_freq, npad="auto")

    if notch_freqs is not None:
      raw.notch_filter(notch_freqs)

    events, _ = mne.events_from_annotations(raw,event_id = event_dict, verbose=False)
    picks = mne.pick_channels_regexp(raw.ch_names, regexp=r'EEG*')
    if Bands is not None:
      # Getting epochs for different frequncy bands   
      for band, (fmin, fmax) in enumerate(Bands):
        raw_bandpass = raw.copy().filter(fmin, fmax, fir_design='firwin')
        epochs = mne.Epochs(raw_bandpass, events, event_id=event_dict,baseline=Baseline, tmin=tmin, tmax=tmax, preload=True, picks=picks, event_repeated = 'drop',verbose=False)
        if not 'bands_data' in locals():
          D_SZ = epochs.get_data().shape
          bands_data = np.empty((D_SZ[0],D_SZ[2],D_SZ[1],len(Bands)))
        # Swapping dimensions from (epoch, channel, sample) to (epoch, sample, channel)
        bands_data[:,:,:,band] = epochs.get_data().transpose(0,2,1) 
        del raw_bandpass
    else:
      epochs = mne.Epochs(raw, events, event_id=event_dict,baseline=Baseline, tmin=tmin, tmax=tmax, preload=True, picks=picks, event_repeated = 'drop',verbose=False)
      bands_data = epochs.get_data()
      SZ = bands_data.shape
      # Swapping dimensions from (epoch, channel, sample) to (epoch, sample, channel)
      bands_data = bands_data.transpose(0,2,1).reshape(SZ[0],SZ[2],SZ[1],1)
    
    # reading event type from .mat files 
    mat_vals = scipy.io.loadmat(f'{data_path}A{sub:02d}{dataset_type}.mat')
    tmp_y = mat_vals['classlabel'].flatten()


    # Adjusting events numbers to be compatible with output classes numbers
    tmp_y = tmp_y - 1

    max_epochs = 288
    SZ = bands_data.shape      
    # Creating output x and y matrices
    if not 'data_x' in locals():
      data_x = np.empty((max_epochs*len(Subjects),SZ[1],SZ[2],SZ[3]),dtype=Data_type)
      data_y = np.empty(max_epochs*len(Subjects),dtype=np.uint16)

    ## adjusting data type
    bands_data = bands_data.astype(Data_type)
    tmp_y = tmp_y.astype(np.uint16)

  
    data_x[data_count:data_count + SZ[0],:,:,:] = bands_data
    data_y[data_count:data_count + SZ[0]] = tmp_y
      
    data_count+=SZ[0]

    del raw, epochs, tmp_y, bands_data # saving memory 

  # data_x = data_x[0:data_count,:,:,:]
  # data_y = data_y[0:data_count]


  return data_x, data_y, data_index

########################### Function End ####################################


############################ Function Start #################################
def get_data_physionet(Subjects=np.arange(1,110), Exclude=np.array([88,89,92,100,104]), data_path='/content/physionet/',Bands=None, resample_freq=None, Total=100, Data_type=np.float32, tmin=0, tmax=2,
                       Tasks=np.array([[3,7,11],[5,9,13]]),  Baseline=None, notch_freqs=None):

  # First subject in Subjects and Exclude has number = 1

  if Exclude is not None:
    print('Excluded Subjects are '+ str(Exclude))
    Subjects = np.delete(Subjects,np.isin(Subjects,Exclude))
  else:
    print('No subject excluded')

  subject_count = 0
  data_index = np.zeros((len(Subjects),2)).astype(np.int) # starting data index for each subject and its number
  data_index[:,1] = Subjects
  data_count = 0
  for sub_index, subject in enumerate(Subjects):
    print(f'\rLoading subject {sub_index+1}/{len(Subjects)}', end = '')
    data_index[subject_count,0] = data_count
    subject_count+=1
    for run in Tasks.flatten():
      file_name = f'{data_path}/files/S{subject:03d}/S{subject:03d}R{run:02d}.edf'
      raw = mne.io.read_raw_edf(file_name,verbose=False)
      raw.load_data() # needed for filteration
      if resample_freq is not None:
        raw.resample(resample_freq, npad="auto")
      if notch_freqs != None:
        raw.notch_filter(notch_freqs)
      events, event_dict = mne.events_from_annotations(raw,verbose=False)
      if Bands is not None:
        # Getting epochs for different frequncy bands   
        for band, (fmin, fmax) in enumerate(Bands):
          raw_bandpass = raw.copy().filter(fmin, fmax, fir_design='firwin')
          epochs = mne.Epochs(raw_bandpass, events, event_id=event_dict,baseline=Baseline, tmin=tmin, tmax=tmax, preload=True, event_repeated = 'drop',verbose=False)
          if not 'bands_data' in locals():
            D_SZ = epochs.get_data().shape
            bands_data = np.empty((D_SZ[0],D_SZ[2],D_SZ[1],len(Bands)))
          # Swapping dimensions from (epoch, channel, sample) to (epoch, sample, channel)
          bands_data[:,:,:,band] = epochs.get_data().transpose(0,2,1) 
          del raw_bandpass
      else:
        epochs = mne.Epochs(raw, events, event_id=event_dict,baseline=Baseline, tmin=tmin, tmax=tmax, preload=True, event_repeated = 'drop',verbose=False)
        bands_data = epochs.get_data()
        SZ = bands_data.shape
        # Swapping dimensions from (epoch, channel, sample) to (epoch, sample, channel)
        bands_data = bands_data.transpose(0,2,1).reshape(SZ[0],SZ[2],SZ[1],1)
      tmp_y = epochs.events[:,2]

      # Adjusting events numbers to be compatible with output classes numbers
      if run in Tasks[0]:
        tmp_y = tmp_y - 1
      elif run in Tasks[1]:
        tmp_y = tmp_y - 1
        tmp_y[tmp_y==1]=3
        tmp_y[tmp_y==2]=4

      SZ = bands_data.shape

      # Creating output x matrix
      if not 'data_x' in locals():
        file_count = 6
        max_epochs = 30
        data_x = np.empty((max_epochs*len(Subjects)*file_count,SZ[1],SZ[2],SZ[3]),dtype=Data_type)
        data_y = np.empty((max_epochs*len(Subjects)*file_count),dtype=np.uint16)

      ## adjusting data type
      bands_data = bands_data.astype(Data_type)
      tmp_y = tmp_y.astype(np.uint16)


      data_x[data_count:data_count + SZ[0],:,:,:] = bands_data
      data_y[data_count:data_count + SZ[0]] = tmp_y
      data_count+=SZ[0]
      del raw, epochs, tmp_y, bands_data # saving memory 
      
      

  data_x = data_x[0:data_count,:,:,:]
  data_y = data_y[0:data_count]

  # removing samples with odd output
  min_cat = 0
  max_cat = 4

  idx = np.flatnonzero(np.logical_or(data_y > max_cat,data_y < min_cat))
  data_x = np.delete(data_x,idx, axis=0)
  data_y = np.delete(data_y,idx)
  del idx

  return data_x, data_y, data_index

########################### Function End ####################################

########################### Function Start ##################################
def get_pos_map(dataset):
  if dataset=='bcicomptIV2a':
    pos_map = np.array([
      [-1,-1,-1, 1,-1,-1,-1], 
      [-1, 2, 3, 4, 5, 6,-1], 
      [ 7, 8, 9,10,11,12,13], 
      [-1,14,15,16,17,18,-1], 
      [-1,-1,19,20,21,-1,-1], 
      [-1,-1,-1,22,-1,-1,-1]])
  elif dataset=='physionet':
    pos_map = np.array([
      [-1,-1,-1,-1,22,23,24,-1,-1,-1,-1],
      [-1,-1,-1,25,26,27,28,29,-1,-1,-1], 
      [-1,30,31,32,33,34,35,36,37,38,-1], 
      [-1,39, 1, 2, 3, 4, 5, 6, 7,40,-1], 
      [43,41, 8, 9,10,11,12,13,14,42,44], 
      [-1,45,15,16,17,18,19,20,21,46,-1], 
      [-1,47,48,49,50,51,52,53,54,55,-1], 
      [-1,-1,-1,56,57,58,59,60,-1,-1,-1], 
      [-1,-1,-1,-1,61,62,63,-1,-1,-1,-1],
      [-1,-1,-1,-1,-1,64,-1,-1,-1,-1,-1]])
  elif dataset=='ttk':
    pos_map = np.array([
      [-1,-1,-1,-1, 1,-1,31,-1,-1,-1,-1],
      [-1,-1,-1,32,33,34,61,60,-1,-1,-1],
      [-1, 4,36, 3,35, 2,62,29,59,30,-1],
      [ 5,37, 6,38, 7,63,28,57,27,58,26],
      [-1, 9,40, 8,39,-1,56,24,55,25,-1],
      [10,41,11,42,12,52,23,53,22,54,21],
      [-1,15,44,14,43,13,51,19,50,20,-1],
      [-1,-1,-1,45,46,47,48,49,-1,-1,-1],
      [-1,-1,-1,-1,16,17,18,-1,-1,-1,-1]])
  else:
    sys.exit("Position map not defined for this dataset.")
  return pos_map
########################### Function End  ###################################

########################### Function Start ##################################

def make_into_2d(data_1d,pos_map):
  Map = pos_map
  map_sz = Map.shape
  SZ = data_1d.shape # (epochs, time samples, eeg channels, bands)
  Map = Map.flatten()
  idx = np.arange(Map.shape[0])
  idx = idx[Map > 0]
  Map = Map[Map > 0]
  Map = Map -1 # adjusting index to start from 0
  data_2d = np.zeros((SZ[0],SZ[1], map_sz[0]*map_sz[1], SZ[3]),dtype=data_1d.dtype)
  
  data_2d[:,:,idx,:] = data_1d[:,:,Map,:]
  data_2d = data_2d.reshape(SZ[0],SZ[1], map_sz[0],map_sz[1], SZ[3])
  return data_2d
########################### Function End ####################################


############################ Function Start #################################
def unzip_dataset(dataset):
  if dataset=='physionet':
    output_path = '/content/physionet/'
    zipfile_list = ['/gdrive/MyDrive/datasets/eeg-motor-movementimagery-dataset-1.0.0.zip']
  elif dataset=='bcicomptIV2a':
    output_path = '/content/bcicomptIV2a/'
    zipfile_list = ['/gdrive/MyDrive/datasets/BCICIV_2a_gdf.zip', '/gdrive/MyDrive/datasets/true_labels.zip']
  elif dataset=='ttk':
    print(f'Dataset ttk has no zip files')
    return
  else:
    raise ValueError('Unknown dataset')
  if os.path.exists(output_path):
    print('Data directory already exists.')
    return

  print ('Unzipping data.')
  for zipfile in zipfile_list:
    get_ipython().system('unzip -qq ' + zipfile +' -d ' + output_path)
  print ('Unzipping done.')  
  return
########################### Function End ####################################


########################### Function Start ##################################
def balance(data_x, data_y, data_index):
  # make number of trial for each subject equal to the minimum per class
  data_index = data_index.copy() # to avoid changing the original array
  Classes, counts = (),()
  end=len(data_y)
  for sub_index in reversed(range(len(data_index))):
    start=data_index[sub_index,0]
    val, cnt = np.unique(data_y[start:end], return_counts=True)
    Classes=(val,*Classes)
    counts = (cnt,*counts)
    end=start
  MIN = np.array(counts).min(axis=1)

  for Class in Classes[0]:
    index=np.flatnonzero(data_y==Class)
    end=len(data_y)
    index1=np.array([]).astype(np.int)
    for sub_index in reversed(range(len(data_index))):
      start=int(data_index[sub_index,0])
      diff = int(counts[sub_index][Classes[sub_index]==Class]-MIN[sub_index])
      index1 = np.append(index[index>=start][0:diff], index1)
      data_index[sub_index+1:,0] -= diff
      end=start
    
    data_x = np.delete(data_x,index1,axis=0)
    data_y = np.delete(data_y,index1,axis=0)

    print ('Balancing data done.')

    return data_x, data_y, data_index
############################ Function End ###################################


############################ Function Start #################################
def normalize(data_x):
  """
  Normalize the data (for each band) to have zero mean and unity standard deviation
  works in place
  """
  SZ = data_x.shape
  for i in range(SZ[-1]):
    if len(SZ) == 4:
      mean = np.mean(data_x[:,:,:,i])
      std = np.std(data_x[:,:,:,i])
      data_x[:,:,:,i] -= mean
      data_x[:,:,:,i] /= std
    elif len(SZ) == 5:
      mean = np.mean(data_x[:,:,:,:,i])
      std = np.std(data_x[:,:,:,:,i])
      data_x[:,:,:,:,i] -= mean
      data_x[:,:,:,:,i] /= std
    else:
      raise ValueError('data_x has unexpected size')
      
  print ('Normalizing data done.')
  return

############################ Function End ###################################


############################ Function Start #################################
def video_array(data_x, data_y, Class=0, Band=0,Rows=4, Cols=5, Seed=100):

  import matplotlib.pyplot as plt
  import matplotlib.animation as animation
  from IPython.display import HTML

  fig = plt.figure()
  ims = []
  index = np.flatnonzero(data_y == Class)
  np.random.seed(Seed)
  np.random.shuffle(index)
  index = index[:Rows*Cols]
  print('samples = ' +str(index))
  total_frames = data_x.shape[1]
  for frame in range(total_frames):
    img_count = 0
    img = data_x[index[img_count],frame,:,:,Band]
    img_count+=1
    for col in range(1,Cols):
      img = np.append(img,data_x[index[img_count],frame,:,:,Band],axis=1)
      img_count+=1
    for row in range(1,Rows):
      img_col = data_x[index[img_count],frame,:,:,Band]
      img_count+=1
      for col in range(1,Cols):
        img_col = np.append(img_col,data_x[index[img_count],frame,:,:,Band],axis=1)
        img_count+=1
      img = np.append(img,img_col,axis=0)

    im = plt.imshow(img, animated=True)
    plt.clim(-2, 2)  
    ims.append([im])

  plt.colorbar()
  plt.xticks([])
  plt.yticks([])
  plt.close()
  ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)
  display(HTML(ani.to_html5_video()))
  return
############################# Function End ##################################

############################ Function Start #################################
def remove_classes(data_x, data_y, Classes, data_index):
  data_index = data_index.copy()
  CLS, counts = (),()
  end=len(data_y)
  for sub_index in reversed(range(len(data_index))):
    start=data_index[sub_index,0]
    val, cnt = np.unique(data_y[start:end], return_counts=True)
    CLS=(val,*CLS)
    counts = (cnt,*counts)
    end=start
  CLS = CLS[0]
  counts = np.cumsum(np.transpose(np.array(counts)),axis=1)
  for cls in Classes:
    print(counts[np.where(CLS==cls),0:-1].shape)
    data_index[1:,0] = data_index[1:,0] - counts[np.where(CLS==cls),0:-1]
    
  data_x = np.delete(arg,np.isin(data_y,Classes),axis=0)
  data_y = np.delete(data_y,np.isin(data_y,Classes),axis=0)
  return data_x, data_y, data_index
############################ Function End ###################################

############################ Function Start #################################
def remove_subjects(data_index, Subjects, *argv):
  data_index=data_index.copy()
  index=np.ones(len(argv[0]), dtype=bool)

  for sub_index in Subjects:
    start = data_index[sub_index,0]
    if sub_index == len(data_index)-1:
      end = None
    else:
      end = data_index[sub_index + 1,0]
    index[start:end]=False

  out = ()
  for arg in argv:
    out += (arg[index],)
  

  for sub_index in Subjects:
    start=data_index[sub_index,0]
    if sub_index==len(data_index)-1:
      end=len(argv[0])
    else:
      end=data_index[sub_index+1,0]
    data_index[(sub_index+1):,0] -= (end-data_index[sub_index,0])
  
  data_index = np.delete(data_index,Subjects, axis=0)

  out=(data_index, *out)
  return out
############################ Function End ###################################

############################ Function Start #################################
def shuffle_subjects(data_x,data_y,data_index,seed=100):

  data_index = data_index.copy()

  data_length = np.append(data_index[1:,0],data_x.shape[0])  - data_index[:,0]


  np.random.seed(seed)
  np.random.shuffle(data_length)

  np.random.seed(seed)
  np.random.shuffle(data_index)

  data_x_out = np.zeros(data_x.shape)
  data_y_out = np.zeros(data_y.shape)

  start_out = 0
  for c  in range(len(data_index)):
    end_out = start_out + data_length[c]
    start_in = data_index[c,0]
    end_in = start_in + data_length[c]
    data_x_out[start_out:end_out] = data_x[start_in:end_in]
    data_y_out[start_out:end_out] = data_y[start_in:end_in]

    data_index[c,0] = start_out
    start_out = end_out
  
  return data_x, data_y, data_index
############################ Function End ###################################



############################ Function Start #################################
def build_model(data_x, data_yc, model_type='CNN1D', show_summary=True, batch_norm=False,dropout_rate=0.5):

  if isinstance(data_x,list):
    input_list_SZ = len(data_x)
    data_x = data_x[0]
  else:
    input_list_SZ = 1

    
  SZ = data_x.shape

  if len(SZ)== 4:
    x_dim = SZ[2]
    y_dim = 1
    n_bands = SZ[3]
  elif len(SZ)==5:
    x_dim = SZ[2]
    y_dim = SZ[3]
    n_bands = SZ[4]

  n_timesteps, n_outputs = SZ[1], data_yc.shape[1]


  if model_type=='EEGNet':
    nb_classes = n_outputs
    Chans=x_dim*y_dim
    Samples=n_timesteps
    dropoutRate=0.5
    kernLength=64
    F1=8
    D=2
    F2=16
    norm_rate=0.25
    dropoutType=Dropout
    #dropoutType=SpatialDropout2D # another option for dropout
    
    input_shape = (Samples, Chans, n_bands)
    conv_filters = (kernLength, 1)
    depth_filters = (1, Chans)
    pool_size = (6, 1)
    pool_size2 = (12, 1)
    separable_filters = (20, 1)
    axis = -1

    input1 = Input(shape=input_shape)
    block1 = Conv2D(F1, conv_filters, padding='same', input_shape=input_shape,use_bias=False)(input1)
    if batch_norm: block1 = BatchNormalization(axis=axis)(block1)
    block1 = DepthwiseConv2D(depth_filters, use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
    if batch_norm: block1 = BatchNormalization(axis=axis)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D(pool_size)(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, separable_filters, use_bias=False, padding='same')(block1)
    if batch_norm: block2 = BatchNormalization(axis=axis)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D(pool_size2)(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    model = Model(inputs=input1, outputs=softmax)
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    if show_summary:
      model.summary()
    return model


  if model_type=='ShallowConvNet':
    ########### need these for ShallowConvNet
    def square(x):
        return K.square(x)
    def log(x):
        return K.log(K.clip(x, min_value=1e-7, max_value=10000))
    ######################################

    nb_classes = n_outputs
    Chans=x_dim*y_dim
    Samples=n_timesteps
    dropoutRate=0.5

    
    input_shape = (Samples, Chans, n_bands)
    conv_filters = (25, 1)
    conv_filters2 = (1, Chans)
    pool_size = (45, 1)
    strides = (15, 1)
    axis = -1


    input_main = Input(input_shape)
    block1 = Conv2D(20, conv_filters, input_shape=input_shape, kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(20, conv_filters2, use_bias=False, kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    if batch_norm: block1 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=pool_size, strides=strides)(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    model = Model(inputs=input_main, outputs=softmax)
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    if show_summary:
      model.summary()
    return model

  if model_type=='DeepConvNet':
    nb_classes = n_outputs
    Chans=x_dim*y_dim
    Samples=n_timesteps
    dropoutRate=0.5
    
    input_shape = (Samples, Chans, n_bands)
    conv_filters = (2, 1)
    conv_filters2 = (1, Chans)
    pool = (2, 1)
    strides = (2, 1)
    axis = -1


    # start the model
    input_main = Input(input_shape)
    block1 = Conv2D(25, conv_filters, input_shape=input_shape, kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(25, conv_filters2, kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    if batch_norm: block1 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=pool, strides=strides)(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, conv_filters, kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    if batch_norm: block2 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=pool, strides=strides)(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, conv_filters, kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    if batch_norm: block3 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=pool, strides=strides)(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, conv_filters, kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    if batch_norm: block4 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=pool, strides=strides)(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)

    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    model = Model(inputs=input_main, outputs=softmax)
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    if show_summary:
      model.summary()
    return model


  if model_type=='EEGNet_fusion':
    """
    The author of this model is Karel Roots and was published along with the paper titled 
    "Fusion Convolutional Neural Network for Cross-Subject EEG Motor Imagery Classification"
    """

    nb_classes = n_outputs
    Chans=x_dim*y_dim
    Samples=n_timesteps
    dropoutRate=0.5
    norm_rate=0.25
    dropoutType=Dropout
    #dropoutType=SpatialDropout2D # another option for dropout
    
    input_shape = (Samples, Chans, n_bands)
    conv_filters = (64, 1)
    conv_filters2 = (96, 1)
    conv_filters3 = (128, 1)    
    #depth_filters = (1, Chans)
    depth_filters = (n_bands, Chans) # made improvement
    pool_size = (4, 1)
    pool_size2 = (8, 1)
    separable_filters = (8, 1)
    separable_filters2 = (16, 1)
    separable_filters3 = (32, 1)
    axis = -1

    F1 = 8
    F1_2 = 16
    F1_3 = 32
    F2 = 16
    F2_2 = 32
    F2_3 = 64
    D = 2
    D2 = 2
    D3 = 2

    input1 = Input(shape=input_shape)
    block1 = Conv2D(F1, conv_filters, padding='same', input_shape=input_shape, use_bias=False)(input1)
    if batch_norm: block1 = BatchNormalization(axis=axis)(block1)
    block1 = DepthwiseConv2D(depth_filters, use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
    if batch_norm: block1 = BatchNormalization(axis=axis)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D(pool_size)(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, separable_filters, use_bias=False, padding='same')(block1)  # 8
    if batch_norm: block2 = BatchNormalization(axis=axis)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D(pool_size2)(block2)
    block2 = dropoutType(dropoutRate)(block2)
    block2 = Flatten()(block2)  # 13

    # 8 - 13

    input2 = Input(shape=input_shape)
    block3 = Conv2D(F1_2, conv_filters2, padding='same', input_shape=input_shape, use_bias=False)(input2)
    if batch_norm: block3 = BatchNormalization(axis=axis)(block3)
    block3 = DepthwiseConv2D(depth_filters, use_bias=False, depth_multiplier=D2, depthwise_constraint=max_norm(1.))(block3)
    if batch_norm: block3 = BatchNormalization(axis=axis)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D(pool_size)(block3)
    block3 = dropoutType(dropoutRate)(block3)

    block4 = SeparableConv2D(F2_2, separable_filters2, use_bias=False, padding='same')(block3)  # 22
    if batch_norm: block4 = BatchNormalization(axis=axis)(block4)
    block4 = Activation('elu')(block4)
    block4 = AveragePooling2D(pool_size2)(block4)
    block4 = dropoutType(dropoutRate)(block4)
    block4 = Flatten()(block4)  # 27
    # 22 - 27

    input3 = Input(shape=input_shape)
    block5 = Conv2D(F1_3, conv_filters3, padding='same', input_shape=input_shape, use_bias=False)(input3)
    if batch_norm: block5 = BatchNormalization(axis=axis)(block5)
    block5 = DepthwiseConv2D(depth_filters, use_bias=False, depth_multiplier=D3, depthwise_constraint=max_norm(1.))(block5)
    if batch_norm: block5 = BatchNormalization(axis=axis)(block5)
    block5 = Activation('elu')(block5)
    block5 = AveragePooling2D(pool_size)(block5)
    block5 = dropoutType(dropoutRate)(block5)

    block6 = SeparableConv2D(F2_3, separable_filters3, use_bias=False, padding='same')(block5)  # 36
    if batch_norm: block6 = BatchNormalization(axis=axis)(block6)
    block6 = Activation('elu')(block6)
    block6 = AveragePooling2D(pool_size2)(block6)
    block6 = dropoutType(dropoutRate)(block6)
    block6 = Flatten()(block6)  # 41

    # 36 - 41

    merge_one = concatenate([block2, block4])
    merge_two = concatenate([merge_one, block6])

    flatten = Flatten()(merge_two)

    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(flatten)

    softmax = Activation('softmax', name='softmax')(dense)

    model= Model(inputs=[input1, input2, input3], outputs=softmax)

    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    if show_summary:
      model.summary()
    return model

  if model_type=='CNN1D_MFBF':

    input_set=[]
    block_set=[]
    dropoutRate = dropout_rate
    norm_rate=0.25
    
    for band in range(input_list_SZ):
      input_set.append(Input(shape=(n_timesteps, x_dim*y_dim)))
      #block1 = Lambda(lambda x: x[:,:,:,band])(input_set[band])
      
      block1 = AveragePooling1D(pool_size=(2))(input_set[band])
      block1 = Conv1D(50,5,padding='same')(block1)
      block1 = Activation('elu')(block1) # try elu instead of relu
      block1 = Dropout(dropoutRate)(block1)
      
      block1 = AveragePooling1D(pool_size=(2))(block1)
      block1 = Conv1D(50,5,padding='same')(block1)
      block1 = Activation('elu')(block1) # try elu instead of relu
      block1 = Dropout(dropoutRate)(block1)
      
      block1 = AveragePooling1D(pool_size=(2))(block1)
      block1 = Conv1D(50,5,padding='same')(block1)
      block1 = Activation('elu')(block1) # try elu instead of relu
      block1 = Dropout(dropoutRate)(block1)

      block1 = AveragePooling1D(pool_size=(2))(block1)
      block1 = Dropout(dropoutRate)(block1)
      block_set.append(Flatten()(block1))

    # merging models  
    merge_block  = concatenate(block_set)
    flatten = Flatten()(merge_block)
    dense = Dense(n_outputs, name='dense', kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)
    
    model= Model(inputs=input_set, outputs=softmax)

    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    if show_summary:
      model.summary()
    return model

  # Adding separate model.add(Activation(activations.relu)) line
  # helps with memeory leak
  # see link: https://github.com/tensorflow/tensorflow/issues/31312
  model = Sequential()
  if model_type=='Basic':
    model.add(InputLayer(input_shape=(n_timesteps, x_dim * y_dim, n_bands)))
    model.add(Flatten())
  elif model_type=='CNN1D':
    model.add(InputLayer(input_shape=(n_timesteps, x_dim * y_dim, n_bands)))
    model.add(Reshape((n_timesteps, x_dim*y_dim*n_bands)) )

    model.add(AveragePooling1D(pool_size=(5)))
    
    model.add(Conv1D(50, 5,  padding='same', activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
    model.add(AveragePooling1D(pool_size=(2)))
    model.add(Conv1D(50, 5, padding='same', activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
    model.add(AveragePooling1D(pool_size=(2)))
    model.add(Conv1D(50, 5, padding='same', activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(100, activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))

  elif model_type=='CNN2D':
    model.add(InputLayer(input_shape=(n_timesteps, x_dim , y_dim, n_bands)))
    model.add(Reshape((n_timesteps, x_dim*y_dim,n_bands)) )
    model.add(AveragePooling2D(pool_size=(5,1)))
    model.add(Conv2D(50, (5,5),  padding='same', activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
    model.add(AveragePooling2D(pool_size=(2,1)))
    model.add(Conv2D(50, (5,5),  padding='same', activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
    model.add(AveragePooling2D(pool_size=(2,1)))
    model.add(Conv2D(50, (5,2),  padding='same', activation=None))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(100, activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
  elif model_type=='CNN3D':
    model.add(InputLayer(input_shape=(n_timesteps, x_dim , y_dim, n_bands)))
    model.add(AveragePooling3D(pool_size=(2,1,1)))
    model.add(Conv3D(50, (5,2,2),  padding='same', activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
    model.add(AveragePooling3D(pool_size=(2,1,1)))
    model.add(Conv3D(50, (5,2,2),  padding='same', activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
    model.add(AveragePooling3D(pool_size=(2,1,1)))
    model.add(Conv3D(50, (5,2,2),  padding='same', activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(100, activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
  elif model_type=='TimeDist': # Time distributed
    model.add(InputLayer(input_shape=(n_timesteps, x_dim , y_dim, n_bands)))
    model.add(AveragePooling3D(pool_size=(5,1,1))) 
    model.add(TimeDistributed(Conv2D(50, (3, 3), strides=(1, 1), activation=None, padding='same')))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
    model.add(TimeDistributed(Conv2D(50, (2, 2), strides=(1, 1), activation=None, padding='same')))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(200, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(dropout_rate))
  

  model.add(Dense(n_outputs, activation='softmax'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  if show_summary:
    model.summary()

  return model
######################### Function End ######################################


######################### Function Start ####################################
def split_data(data, Splits=5, split_index=0, model_type=None, shuffle=False, shuffle_seed=100):
  # specify model_type when spliting x data to adjust the output according the the model

  data=data.copy()

  if shuffle:
    np.random.seed(shuffle_seed)
    np.random.shuffle(data)
    
  SZ = data.shape
  split_length = np.floor(SZ[0]/Splits).astype(np.int)
  start = split_length*split_index
  end = start + split_length
  if split_index == Splits-1:
    end = SZ[0]



  index = np.ones(data.shape[0],dtype=bool)
  index[start:end] = False

  train = data[index]
  index = np.invert(index)
  test = data[index]

  # adjusting output depending on the model

  if model_type=='EEGNet_fusion':
    train=[train, train, train]
    test=[test, test, test]

  if model_type=='CNN1D_MFBF':
    Bands=data[0].shape[-1]
    train=[np.expand_dims(train[:,:,:,i],axis=3) for i in range(Bands)]
    test=[np.expand_dims(test[:,:,:,i],axis=3) for i in range(Bands)]


  return train, test

########################## Function End #####################################

######################### Function Start ####################################
def split_subjects(data,data_index,test_subject_index, model_type=None):

  test_subject_index = np.array([test_subject_index]).flatten()
  test_index = np.zeros(len(data), dtype=bool)
  
  for sub_index in test_subject_index:
    start = data_index[sub_index,0]
    if sub_index == len(data_index)-1:
      end=None
    else:
      end=data_index[sub_index+1,0]
    test_index[start:end] = True 

  test = data[test_index]
  train = data[np.invert(test_index)]

  # adjusting output depending on the model

  if model_type=='EEGNet_fusion':
    train=[train, train, train]
    test=[test, test, test]

  if model_type=='CNN1D_MFBF':
    Bands=data[0].shape[-1]
    train=[np.expand_dims(train[:,:,:,i],axis=3) for i in range(Bands)]
    test=[np.expand_dims(test[:,:,:,i],axis=3) for i in range(Bands)]
  return train, test
########################## Function End #####################################


######################### Function Start ####################################
def evaluate_model(model_list,dataset,Bands,data_x, data_y, data_index,splits,Iterations, epochs, batch_size=64, verbose=0, 
show_summary = False,batch_norm=True,dropout_rate=0.5, align_to_subject=True, shuffle=True, play_audio=False):

  results = {}
  time_results = {}
  if align_to_subject:
    subject_results = {}
  else:
    subject_results = None
  for model_type in model_list:
    print(f'Start evaluation for {model_type} model at {datetime.now().strftime("%H:%M:%S")} UTC time. ({dataset}, {len(Iterations)} iterations ,Bands {"off" if not Bands else "on"})')
    if splits > data_index.shape[0] and align_to_subject:
      raise ValueError('Number of splits should be less than number of subjects.')
    elif splits>data_x.shape[0]:
      raise ValueError('Number of splits should be less than number of trials.')

    if model_type in ['CNN3D', 'TimeDist']: # these model requires 2D mapped data
      # Generating 2D mapped data
      pos_map = get_pos_map(dataset) # positions for 2D map conversion
      data_input = make_into_2d(data_x,pos_map)
    else:
      data_input = data_x

    if align_to_subject:# distributing subjects on splits in a fair way
      N = data_index.shape[0]
      sub_split = np.ones(splits)*np.floor(N/splits)
      sub_split[0:(N % splits)] +=1
      sub_split_index = np.append(0,np.cumsum(sub_split)).astype(np.int)

    model_data=[]
    model_time_data=[]
    if align_to_subject:
      model_subject_data=[]

    for iteration in Iterations:
      start_time = time.time()
      # Splitting data 
      if align_to_subject:
        sub_test_index = np.arange(sub_split_index[iteration],sub_split_index[iteration+1])
        train_x, test_x = split_subjects(data_x,data_index,sub_test_index,model_type=model_type)
        train_y, test_y = split_subjects(data_y,data_index,sub_test_index,model_type=None)
        print(f'Test subjects for split No. {iteration+1}: {data_index[sub_test_index,1]}')
      else:
        train_x, test_x = split_data(data_input, Splits=splits, split_index=iteration, shuffle=shuffle,model_type=model_type)
        train_y, test_y = split_data(data_y, Splits=splits, split_index=iteration, shuffle=True,model_type=None)

      # Generating categorical data
      train_yc = to_categorical(train_y)
      test_yc = to_categorical(test_y)

      # Building and validating  model
      model = build_model(train_x, train_yc, model_type=model_type,show_summary=show_summary, batch_norm=batch_norm,dropout_rate=dropout_rate)    
      model.fit(train_x, train_yc, epochs=epochs, batch_size=batch_size, verbose=verbose,  validation_data=(test_x, test_yc))

      # fixing memory leak
      K.clear_session()
      tf.compat.v1.reset_default_graph() # TF graph isn't same as Keras graph
      gc.collect()

      model_data.append(model.history.history)
      #iteration_time = timedelta(seconds=round(time.time() - start_time))   
      iteration_time = time.time() - start_time
      model_time_data.append(iteration_time)
      if align_to_subject:
        model_subject_data.append(data_index[sub_test_index,1])

      # saving partial results
      results[model_type]=model_data
      time_results[model_type]=model_time_data
      subject_results[model_type] = model_subject_data
      save_results(results,time_results,subject_results, dataset, Bands,model_type=model_type,partial=True)

      print(f'Split No. {iteration+1}. done in {timedelta(seconds=round(iteration_time))} ')
    
    # saving final resutls for model
    results[model_type]=model_data
    time_results[model_type]=model_time_data
    subject_results[model_type] = model_subject_data
    save_results(results,time_results,subject_results, dataset, Bands,model_type=model_type)


  

  # Play an audio to alert for finishing
  if play_audio:
    from google.colab import output
    output.eval_js('new Audio("https://upload.wikimedia.org/wikipedia/commons/4/42/Bird_singing.ogg").play()')

  return model, results, time_results,  (subject_results if align_to_subject else None)

########################## Function End #####################################

######################### Function Start ###################################
def predict_model(model,model_type, data_x, data_y,data_index,splits,Iterations, align_to_subject=True, shuffle=True):

  if splits > data_index.shape[0] and align_to_subject:
    raise ValueError('Number of splits should be less than number of subjects.')
  elif splits>data_x.shape[0]:
    raise ValueError('Number of splits should be less than number of trials.')

  if model_type in ['CNN3D', 'TimeDist']: # these model requires 2D mapped data
    # Generating 2D mapped data
    pos_map = get_pos_map(dataset) # positions for 2D map conversion
    data_input = make_into_2d(data_x,pos_map)
  else:
    data_input = data_x

  if align_to_subject:# distributing subjects on splits in a fair way
    N = data_index.shape[0]
    sub_split = np.ones(splits)*np.floor(N/splits)
    sub_split[0:(N % splits)] +=1
    sub_split_index = np.append(0,np.cumsum(sub_split)).astype(np.int)

  for iteration in Iterations:
    print(f'Prediction accuracy for split No. {iteration+1}')
    # Splitting data 
    if align_to_subject:
      sub_test_index = np.arange(sub_split_index[iteration],sub_split_index[iteration+1])
      _, test_x = split_subjects(data_x,data_index,sub_test_index,model_type=model_type)
      _, test_y = split_subjects(data_y,data_index,sub_test_index,model_type=None)
    else:
      _, test_x = split_data(data_input, Splits=splits, split_index=iteration, shuffle=shuffle,model_type=model_type)
      _, test_y = split_data(data_y, Splits=splits, split_index=iteration, shuffle=True,model_type=None)


    y_predict = np.argmax(model.predict(test_x), axis = 1)
    accuracy = np.sum(y_predict == test_y)/len(test_y)
    print (f'   All classes: {accuracy*100:0.1f}%')

    for CLS in np.unique(test_y):
      index = np.flatnonzero(test_y==CLS)
      acc = np.sum(y_predict[index] == test_y[index])/len(index)
      print (f'   Class {CLS}: {acc*100:0.1f}%')
  return
########################## Function End ######################################


######################### Function Start #####################################
def divide_time(data_x,data_y,N, data_index=None):
  ax = 1 # this axis is the time samples
  if N not in range(1,21): raise ValueError("N must be between 1 and 20")
  if data_x.shape[ax] % N is not 0: raise ValueError("Time samples in input data array must be multiple of N")
  if N is 1: return data_x, data_y
  
  new_shape = list(data_x.shape)
  new_shape[ax-1] = data_x.shape[ax-1]*N
  new_shape[ax] = int(data_x.shape[ax]/N)
  data_x = np.reshape(data_x, new_shape)

  data_y = np.repeat(data_y,N)

  if data_index != None:
    data_index_new=data_index.copy()
    data_index_new *= N
    return data_x, data_y, data_index
  else:
    return data_x, data_y
########################## Function End ######################################


######################### Function Start #####################################
def plot_results(results):

  for model_type in results.keys():
    init = True
    model_data=results[model_type]
    for split_index,split in enumerate(model_data):
      training_loss = split['loss']
      validation_loss = split['val_loss']
      training_acc = split['accuracy']
      validation_acc = split['val_accuracy']
      epochs = range(1, len(training_acc) + 1)


      # plt.figure()
      # plt.plot(epochs, training_loss, 'r--', label='Training loss')
      # plt.plot(epochs, validation_loss, 'b', label='Validation loss')
      # plt.title('Training and validation loss')
      # plt.xlabel('Epochs')
      # plt.ylabel('Loss')
      # plt.legend()


      plt.figure()
      plt.plot(epochs, training_acc, 'r--', label='Training accurecy')
      plt.plot(epochs, validation_acc, 'b', label='Validation accurecy')
      plt.title(f'Accuracy for {model_type} split No. {split_index+1} model')
      plt.xlabel('Epochs')
      plt.ylabel('Accurecy')
      plt.legend()
      plt.show()

      if init:
        training_loss_avg = np.array(training_loss)
        validation_loss_avg = np.array(validation_loss)
        training_acc_avg = np.array(training_acc)
        validation_acc_avg = np.array(validation_acc)
        init=False
      else:
        training_loss_avg += training_loss
        validation_loss_avg += validation_loss
        training_acc_avg += training_acc
        validation_acc_avg += validation_acc
    



    # plotting average values 
    splits_num=len(model_data)
    training_loss_avg /= splits_num
    validation_loss_avg /= splits_num
    training_acc_avg /= splits_num
    validation_acc_avg /= splits_num
    epochs = range(1, len(training_acc_avg) + 1)

    plt.figure()
    plt.plot(epochs, training_acc, 'r--', label='Training accurecy')
    plt.plot(epochs, validation_acc, 'b', label='Validation accurecy')
    plt.title(f'Mean accurecy (for {model_type} model')
    plt.xlabel('Epochs')
    plt.ylabel('Accurecy')
    plt.legend()
    plt.show()

  return
########################## Function End ######################################


######################### Function Start #####################################
def save_results(results,time_results,subject_results,dataset,Bands,model_type=None, partial=False):

  Band_status= 'bandson' if Bands else 'bandsoff' 

  models=[model_type] if model_type else results.keys()
  now = datetime.now().strftime("%Y%m%d_%H%M%S") if not partial else "partial"

  for model_type in models:
    model_data=results[model_type]
    file_h = open(f'/gdrive/MyDrive/results/{dataset}_{model_type}_{Band_status}_{now}.txt', "w")
    writer = csv.writer(file_h)
    for split in model_data:
      d = split['val_accuracy']
      writer.writerow([f'{x:0.4f}' for x in d])  
    file_h.close()

  models=[model_type] if model_type else time_results.keys()

  for model_type in models:
    model_time_data=time_results[model_type]
    file_h = open(f'/gdrive/MyDrive/results/{dataset}_{model_type}_{Band_status}_time_{now}.txt', "w")
    writer = csv.writer(file_h)
    writer.writerow([f'{x:0.2f}' for x in model_time_data])
    file_h.close()

  if subject_results==None:
    print('Data saved')
    return

  models=[model_type] if model_type else subject_results.keys()

  for model_type in models:
    model_subject_data=subject_results[model_type]
    file_h = open(f'/gdrive/MyDrive/results/{dataset}_{model_type}_{Band_status}_subjects_{now}.txt', "w")
    writer = csv.writer(file_h)
    writer.writerow(model_subject_data)
    file_h.close()


  if not partial: print('Data saved')
  return
########################## Function End ######################################
