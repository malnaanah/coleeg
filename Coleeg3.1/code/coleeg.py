__version__ = 3.1

# importing modules
import mne
import numpy as np
from datetime import datetime, timedelta, timezone
import pytz
import scipy.io
from scipy.fft import dct
import sys
import time
import os
from IPython import get_ipython
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Lambda, Permute
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





# Module variables
# this is a pointer to the module object instance itself.
this = sys.modules[__name__]
this.DATA_FOLDER = None
this.RESULT_FOLDER = None
this.CONTENT_FOLDER = None
this.TIME_ZONE = None
this.METRICS = None
this.METRICS_TO_SAVE = None





####################### defining CohenKappa Class ##########################
class CohenKappa(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='cohen_kappa', **kwargs):
        super(CohenKappa, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.conf_mtx = self.add_weight(name='conf_mtx', shape=(num_classes, num_classes), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        confusion = tf.cast(tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes), dtype=tf.float32)
        self.conf_mtx.assign_add(confusion)

    def result(self):
        sum_diag = tf.reduce_sum(tf.linalg.diag_part(self.conf_mtx))
        sum_rows = tf.reduce_sum(self.conf_mtx, axis=0)
        sum_cols = tf.reduce_sum(self.conf_mtx, axis=1)
        total_samples = tf.reduce_sum(sum_rows)

        po = sum_diag / total_samples
        pe = tf.reduce_sum(sum_rows * sum_cols) / (total_samples ** 2)
        kappa = (po - pe) / (1 - pe)

        return kappa

    def reset_states(self):
        # Reset the confusion matrix
        self.conf_mtx.assign(tf.zeros_like(self.conf_mtx))
############################################################################





############################ Function Start ################################
def coleeg_version():
  print(this.__version__)
############################ Function End #################################

############################ Function Start ################################
def set_metrics(metrics,n_outputs=4):
  this.METRICS = []
  this.METRICS_TO_SAVE = metrics
  if any(item in ['accuracy','val_accuracy'] for item in this.METRICS_TO_SAVE):
    this.METRICS.append('accuracy')
  if any(item in ['cohen_kappa','val_cohen_kappa'] for item in this.METRICS_TO_SAVE):
    this.METRICS.append(CohenKappa(num_classes=n_outputs))
  if any(item in ['specificity','val_specificity'] for item in this.METRICS_TO_SAVE):
    this.METRICS.append(specificity)
  if any(item in ['sensitivity','val_sensitivity'] for item in this.METRICS_TO_SAVE):
    this.METRICS.append(sensitivity)
############################ Function End #################################

############################ Function Start ################################
def get_metrics():
  return this.METRICS_TO_SAVE
############################ Function End #################################

############################ Function Start ################################
def set_time_zone(zone):
  this.TIME_ZONE = zone
############################ Function End #################################

############################ Function Start ################################
def set_data_folder(folder):
  from os import path

  this.DATA_FOLDER = folder
  if path.exists(this.DATA_FOLDER) == False: os.mkdir(this.DATA_FOLDER)
  if 'google.colab' in sys.modules:
    this.CONTENT_FOLDER = '/content'
  else:
    this.CONTENT_FOLDER = f'{this.DATA_FOLDER}/content'
    if path.exists(this.CONTENT_FOLDER) == False: os.mkdir(this.CONTENT_FOLDER)

############################ Function End #################################

############################ Function Start ################################
def set_result_folder(folder):
  from os import path
  this.RESULT_FOLDER = f'{this.DATA_FOLDER}/{folder}'
  if path.exists(this.RESULT_FOLDER) == False: os.mkdir(this.RESULT_FOLDER)

############################ Function End #################################

############################ Function Start #################################
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())
############################ Function End #################################

############################ Function Start #################################
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())
############################ Function End #################################


############################ Function Start #################################
def coleeg_info():
  info = """
* Coleeg is an open source initiative for collaborating on making a piece of software for investigating the use of Neural Networks in EEG signal classification on different datasets.

* License GPL V2.0

## Team:
Mahmoud Alnaanah (malnaanah@gmail.com)
Moutz Wahdow (m.wahdow@gmail.com)


## How to install Coleeg to your google drive

  1- From Coleeg github site (https://github.com/malnaanah/coleeg) download the release containing the latest version (currently Coleeg3.1)
  2- Copy Coleeg folder (currently Coleeg3.1) to the root of your google drive.
  3- Run the Colab notebook named RunColeeg.ipynb which is inside Coleeg folder.
  4- Grant permissions for the notebook to enable its access your google drive.
  5- The data needed for Coleeg will be located in the directory ColeegData in the root of your google drive.
  6- To use Colab online, choose Connect>Connect to a hosted runtime.
  7- To use your local machine, download Coleeg folder to your home (i.e. personal) folder.
     Local run time was tested in Linux (kubuntu 22.04).
  8- Run Colab_local script using the command (bash Colab_Local) and copy the generated url.
  9- Run RunColeeg.ipynb from google drive (not the the one on your home folder).
  10- Choose Connect>Connect to a local runtime, then paste the url and select connect.


## How to install dataset files on your Google drive
  * In version 3.1 the required open dataset files are downloaded automatically.

  For pervious versions
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
    
## What's new in Version 3.1
  * Some bugs are fixed
  * Dataset files are downloaded automatically
  * Coleeg folder structure is changed. Coleeg code and data are in separate folders which are Coleeg3.1 and ColeegData.
  * Specific python packages are installed to guarantee compatibility.
## What's new in Version 3.0
  * The code is cleaner and more intuitive
  * Support of local runtime is added, with automatic detection of folders.
    In the home folder ($HOME), coleeg code folder should be place in $HOME/coleeg/ and datasets zip files 
    should be placed in $HOME/datasets, results and content folders are created automatically in the home folder.
  * More metrics are supported and they can be selected during initialization, these metrics are:
    loss, accuracy, cohen_kappa, specificity, sensitivity
  * local timezone can be setl
  * A subset of classes and subjects can be selected for evaluationl
  * Plots of the resutls can be displayed and saved as pdf files in resutls/plots folder.
  * The result average value for last epochs can can be displayed and saved in the results folder.
  * The time for evaluation can be displayed and saved in the results folder
  * Validation can be done for dct and fft transforms of the origianl time signals.



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


Customize Your Keras Metrics (how sensitivity and specificity are defined)
https://medium.com/@mostafa.m.ayoub/customize-your-keras-metrics-44ac2e2980bd
  """
  print(info)
  return
############################ Function End #################################


############################ Function Start #################################
def get_data_ttk(Subjects = np.arange(1,26), Exclude=None, data_path=None,
                        Bands=None, resample_freq=None, Data_type=np.float32, tmin=0, tmax=2,Baseline=None, notch_freqs=None):

  if data_path == None:
    data_path=f'{this.DATA_FOLDER}/datasets/TTK/'

  # First subject in Subjects and Exclude has number = 1

  if Exclude is not None:
    print('Excluded Subjects are '+ str(Exclude))
    Subjects = np.delete(Subjects,np.isin(Subjects,Exclude))
  else:
    print('No subject excluded')

  subject_count = 0
  data_index = np.zeros((len(Subjects),2)).astype(int) # starting data index for each subject and its number
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

    raw.load_data(verbose=False) # needed for filteration
    if resample_freq is not None:
      raw.resample(resample_freq, npad="auto")
    if notch_freqs is not None:
        raw.notch_filter(notch_freqs)
    event_dict = {'Stimulus/S  1':1, 'Stimulus/S  5':2, 'Stimulus/S  7':3, 'Stimulus/S  9':4, 'Stimulus/S 11':5} 
    events, _ = mne.events_from_annotations(raw,event_id = event_dict, verbose=False)
    if Bands is not None:
      # Getting epochs for different frequncy bands   
      for band, (fmin, fmax) in enumerate(Bands):
        raw_bandpass = raw.copy().filter(fmin, fmax, fir_design='firwin',verbose=False)
        epochs = mne.Epochs(raw_bandpass, events, event_id=event_dict,baseline=Baseline, tmin=tmin, tmax=tmax, preload=True, event_repeated = 'drop',verbose=False)
        if not 'bands_data' in locals():
          D_SZ = epochs.get_data(copy=False).shape
          bands_data = np.empty((D_SZ[0],D_SZ[2],D_SZ[1],len(Bands)))
        # Swapping dimensions from (epoch, channel, sample) to (epoch, sample, channel)
        bands_data[:,:,:,band] = epochs.get_data(copy=True).transpose(0,2,1) 
        del raw_bandpass
    else:
      epochs = mne.Epochs(raw, events, event_id=event_dict,baseline=Baseline, tmin=tmin, tmax=tmax, preload=True, event_repeated = 'drop',verbose=False)
      bands_data = epochs.get_data(copy=True)
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
def get_data_bcicomptIV2a(Subjects = np.arange(1,19), Exclude=None, data_path=None,Bands=None, resample_freq=None, Data_type=np.float32, tmin=0, tmax=4,Baseline=None, notch_freqs=None):

  if data_path == None:
    data_path=f'{this.CONTENT_FOLDER}/bcicomptIV2a/'

  # First subject in Subjects and Exclude has number = 1

  if Exclude is not None:
    print('Excluded Subjects are '+ str(Exclude))
    Subjects = np.delete(Subjects,np.isin(Subjects,Exclude))
  else:
    print('No subject excluded')

  subject_count = 0
  data_index = np.zeros((len(Subjects),2)).astype(int) # starting data index for each subject and its number
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
    raw.load_data(verbose=False) # needed for filteration
    if resample_freq is not None:
      raw.resample(resample_freq, npad="auto")

    if notch_freqs is not None:
      raw.notch_filter(notch_freqs)

    events, _ = mne.events_from_annotations(raw,event_id = event_dict, verbose=False)
    picks = mne.pick_channels_regexp(raw.ch_names, regexp=r'EEG*')
    if Bands is not None:
      # Getting epochs for different frequncy bands   
      for band, (fmin, fmax) in enumerate(Bands):
        raw_bandpass = raw.copy().filter(fmin, fmax, fir_design='firwin',verbose=False)
        epochs = mne.Epochs(raw_bandpass, events, event_id=event_dict,baseline=Baseline, tmin=tmin, tmax=tmax, preload=True, picks=picks, event_repeated = 'drop',verbose=False)
        if not 'bands_data' in locals():
          D_SZ = epochs.get_data(copy=False).shape
          bands_data = np.empty((D_SZ[0],D_SZ[2],D_SZ[1],len(Bands)))
        # Swapping dimensions from (epoch, channel, sample) to (epoch, sample, channel)
        bands_data[:,:,:,band] = epochs.get_data(copy=True).transpose(0,2,1) 
        del raw_bandpass
    else:
      epochs = mne.Epochs(raw, events, event_id=event_dict,baseline=Baseline, tmin=tmin, tmax=tmax, preload=True, picks=picks, event_repeated = 'drop',verbose=False)
      bands_data = epochs.get_data(copy=True)
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
def get_data_physionet(Subjects=np.arange(1,110), Exclude=np.array([88,89,92,100,104]), data_path=None,Bands=None, resample_freq=None, Total=100, Data_type=np.float32, tmin=0, tmax=2,Tasks=np.array([[3,7,11],[5,9,13]]),  Baseline=None, notch_freqs=None):


  if data_path == None:
    data_path=f'{this.CONTENT_FOLDER}/physionet/'

  # First subject in Subjects and Exclude has number = 1

  if Exclude is not None:
    print('Excluded Subjects are '+ str(Exclude))
    Subjects = np.delete(Subjects,np.isin(Subjects,Exclude))
  else:
    print('No subject excluded')

  subject_count = 0
  data_index = np.zeros((len(Subjects),2)).astype(int) # starting data index for each subject and its number
  data_index[:,1] = Subjects
  data_count = 0
  for sub_index, subject in enumerate(Subjects):
    print(f'\rLoading subject {sub_index+1}/{len(Subjects)}', end = '')
    data_index[subject_count,0] = data_count
    subject_count+=1
    for run in Tasks.flatten():
      file_name = f'{data_path}/files/S{subject:03d}/S{subject:03d}R{run:02d}.edf'
      raw = mne.io.read_raw_edf(file_name,verbose=False)
      raw.load_data(verbose=False) # needed for filteration
      if resample_freq is not None:
        raw.resample(resample_freq, npad="auto")
      if notch_freqs is not None:
        raw.notch_filter(notch_freqs)
      events, event_dict = mne.events_from_annotations(raw,verbose=False)
      if Bands is not None:
        # Getting epochs for different frequncy bands   
        for band, (fmin, fmax) in enumerate(Bands):
          raw_bandpass = raw.copy().filter(fmin, fmax, fir_design='firwin',verbose=False)
          epochs = mne.Epochs(raw_bandpass, events, event_id=event_dict,baseline=Baseline, tmin=tmin, tmax=tmax, preload=True, event_repeated = 'drop',verbose=False)
          if not 'bands_data' in locals():
            D_SZ = epochs.get_data(copy=False).shape
            bands_data = np.empty((D_SZ[0],D_SZ[2],D_SZ[1],len(Bands)))
          # Swapping dimensions from (epoch, channel, sample) to (epoch, sample, channel)
          bands_data[:,:,:,band] = epochs.get_data(copy=True).transpose(0,2,1) 
          del raw_bandpass
      else:
        epochs = mne.Epochs(raw, events, event_id=event_dict,baseline=Baseline, tmin=tmin, tmax=tmax, preload=True, event_repeated = 'drop',verbose=False)
        bands_data = epochs.get_data(copy=True)
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
  dataset_folder=f'{this.DATA_FOLDER}/datasets'
  if os.path.exists(dataset_folder) == False: os.mkdir(dataset_folder)

  if dataset=='physionet':
    zipfile_list = [f'{this.DATA_FOLDER}/datasets/eeg-motor-movementimagery-dataset-1.0.0.zip']
    zipfile_url_list = ['https://physionet.org/static/published-projects/eegmmidb/eeg-motor-movementimagery-dataset-1.0.0.zip']

    output_path = f'{this.CONTENT_FOLDER}/physionet/'
    
  elif dataset=='bcicomptIV2a':
    zipfile_list = [f'{this.DATA_FOLDER}/datasets/BCICIV_2a_gdf.zip', f'{this.DATA_FOLDER}/datasets/true_labels.zip']
    zipfile_url_list = ['https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip', 'https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip']
    output_path = f'{this.CONTENT_FOLDER}/bcicomptIV2a/'
  elif dataset=='ttk':
    print(f'Dataset ttk has no zip files')
    return
  else:
    raise ValueError('Unknown dataset')

  if os.path.exists(output_path) and len(os.listdir(output_path)) > 0:
    print('Data already exists.')
    return

  for zipfile,url in zip(zipfile_list,zipfile_url_list):
    if not os.path.exists(zipfile):
      print(f'Downloading {zipfile.split("/")[-1]}')
      get_ipython().system(f'wget --no-verbose --show-progress {url} -P {dataset_folder}')


  
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
    index1=np.array([]).astype(int)
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
def keep_classes(data_x, data_y, data_index, selected_classes):

  classes= np.unique(data_y)
  counts = np.zeros((len(data_index),len(classes)),dtype=int)
  end=len(data_x)
  for sub_index in reversed(range(len(data_index))):
    start=data_index[sub_index,0]
    clc_val, clc_count = np.unique(data_y[start:end], return_counts=True)
    clc_index = np.flatnonzero(np.isin(classes,clc_val))
    counts[sub_index,clc_index] = clc_count[clc_index]
    end=start

  clc_index = np.flatnonzero(np.isin(classes,selected_classes))
  data_index_out = data_index.copy()
  data_index_out[:,0]=np.sum(counts[:,clc_index],axis=1)
  selected_index = np.isin(data_y,selected_classes)
  data_x_out = data_x[selected_index]
  data_y_out = data_y[selected_index]
  return data_x_out, data_y_out, data_index_out
############################ Function End ###################################

############################ Function Start #################################
def keep_subjects(data_x, data_y, data_index, selected_subjects):
  # selected_subects: list of subjects to keep, starting from 1 (not 0)
  Subjects = list(data_index[:,1])

  # calculating size of the output data
  sz_out = 0
  selected_indices = np.flatnonzero(np.isin(Subjects,selected_subjects))
  for index in selected_indices:
    if index >=  len(Subjects) - 1:
      sz_out += len(data_x) - data_index[index,0]
    else:
      sz_out += data_index[index + 1,0] - data_index[index,0]

  sz = list(data_x.shape)
  sz[0] = sz_out
  data_x_out = np.empty(sz,dtype=data_x.dtype)
  data_y_out = np.empty(sz[0],dtype=data_y.dtype)
  data_index_out= data_index[selected_indices]

  data_index_out[0,0] = 0
  for i in range(0,len(selected_indices)):
    if i > 0:
      data_index_out[i,0] = data_index_out[i-1,0] + diff
    index = selected_indices[i]
    if index >= len(data_index)-1:
      diff = len(data_x)-data_index[index,0]
    else:
      diff = data_index[index+1,0]-data_index[index,0]




    out_index =data_index_out[i,0]
    in_index = data_index[index,0]
    data_x_out[out_index:out_index+diff] = data_x[in_index:in_index+diff]
    data_y_out[out_index:out_index+diff] = data_y[in_index:in_index+diff]
  return data_x_out, data_y_out, data_index_out
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
def build_model(data_x, data_yc, model_type='CNN1D', show_summary=True, batch_norm=False,apply_spectral=False,dropout_rate=0.5,Max_norm=None):

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

  # update metrics
  set_metrics(this.METRICS_TO_SAVE,n_outputs=n_outputs)




  if model_type=='EEGNet':
    nb_classes = n_outputs
    Chans=x_dim*y_dim
    Samples=n_timesteps
    dropoutRate=0.5
    kernLength=64
    F1=8
    D=2
    F2=16
    dropoutType=Dropout
    #dropoutType=SpatialDropout2D # another option for dropout
    norm_rate=.25
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
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=this.METRICS)
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
    norm_rate=0.5

    
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
    dense = Dense(nb_classes, kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax')(dense)

    model = Model(inputs=input_main, outputs=softmax)
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=this.METRICS)
    if show_summary:
      model.summary()
    return model

  if model_type=='DeepConvNet':
    nb_classes = n_outputs
    Chans=x_dim*y_dim
    Samples=n_timesteps
    dropoutRate=0.5
    norm_rate=0.5
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

    dense = Dense(nb_classes, kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax')(dense)

    model = Model(inputs=input_main, outputs=softmax)
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=this.METRICS)
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
    dropoutType=Dropout
    #dropoutType=SpatialDropout2D # another option for dropout
    norm_rate=0.25
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

    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(0.25))(flatten)

    softmax = Activation('softmax', name='softmax')(dense)

    model= Model(inputs=[input1, input2, input3], outputs=softmax)

    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=this.METRICS)
    if show_summary:
      model.summary()
    return model

  if model_type=='CNN1D_MFBF':

    input_set=[]
    block_set=[]
    dropoutRate = dropout_rate
    
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
    dense = Dense(n_outputs, name='dense', kernel_constraint=max_norm(Max_norm))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)
    
    model= Model(inputs=input_set, outputs=softmax)

    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=this.METRICS)
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
    time_axis = -1
    model.add(InputLayer(input_shape=(n_timesteps, x_dim * y_dim, n_bands)))
    model.add(Reshape((n_timesteps, x_dim*y_dim*n_bands)) )
    # model.add(Permute((2,1), input_shape=(n_timesteps, x_dim * y_dim * n_bands)))
    if batch_norm: model.add(BatchNormalization(axis=time_axis, epsilon=1e-05, momentum=0.1))
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
  
  model.add(Flatten()) # added in Version 3.1 to enhance model validation accuracy
  model.add(Dense(n_outputs, activation='softmax',kernel_constraint=max_norm(Max_norm)))
  model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=this.METRICS)
  if show_summary:
    model.summary()

  return model
######################### Function End ######################################


######################### Function Start ####################################
def split_data(data, Parts=5, part_index=0, model_type=None, shuffle=False, shuffle_seed=100):
  # specify model_type when spliting x data to adjust the output according the the model

  data=data.copy()

  if shuffle:
    np.random.seed(shuffle_seed)
    np.random.shuffle(data)
    
  SZ = data.shape
  part_length = np.floor(SZ[0]/Parts).astype(int)
  start = part_length*part_index
  end = start + part_length
  if part_index == Parts-1:
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
def evaluate_model(model_list,dataset,Bands,data_x, data_y, data_index,fold_num,Folds, epochs, batch_size=64, verbose=0, show_summary = False,batch_norm=True, apply_spectral=False,dropout_rate=0.5, align_to_subject=True, selected_subjects=False,selected_classes=False, shuffle=True, play_audio=False, Max_norm=None):
  tz = pytz.timezone(this.TIME_ZONE)
  metric_results = {}
  time_results = {}
  kappa_results={}


  if selected_classes:
    data_x, data_y, data_index = keep_classes(data_x, data_y, data_index, selected_classes)

  if selected_subjects:
    data_x, data_y, data_index = keep_subjects(data_x, data_y, data_index, selected_subjects)  

  if apply_spectral == 'dct':
    data_x = dct_1d(data_x)
    print('dct applied')
    normalize(data_x)

  if apply_spectral == 'fft':
    data_x = fft_1d(data_x)
    print('fft applied')
    normalize(data_x)



  if align_to_subject:
    subject_results = {}
  else:
    subject_results = None
  for model_type in model_list:
    print(f'Starting evaluation for {model_type} model at {datetime.now(tz).strftime("%H:%M:%S")}. Dataset: {dataset}, Bands: {"off" if not Bands else "on"}, Folds: {list(Folds)}')
    if fold_num > data_index.shape[0] and align_to_subject:
      raise ValueError('Number of folds should be less than number of subjects.')
    elif fold_num>data_x.shape[0]:
      raise ValueError('Number of folds should be less than number of trials.')

    if model_type in ['CNN3D', 'TimeDist']: # these model requires 2D mapped data
      # Generating 2D mapped data
      pos_map = get_pos_map(dataset) # positions for 2D map conversion
      data_input = make_into_2d(data_x,pos_map)
    else:
      data_input = data_x

    if align_to_subject:# distributing subjects on folds in a fair way
      N = data_index.shape[0]
      sub_part = np.ones(fold_num)*np.floor(N/fold_num)
      sub_part[0:(N % fold_num)] +=1
      sub_part_index = np.append(0,np.cumsum(sub_part)).astype(int)

    metric_data=[]
    time_data=[]
    kappa_data=[]

    if align_to_subject:
      subject_data=[]

    for fold in Folds:
      start_time = time.time()
      # Splitting data 
      if align_to_subject:
        sub_test_index = np.arange(sub_part_index[fold],sub_part_index[fold+1])
        train_x, test_x = split_subjects(data_x,data_index,sub_test_index,model_type=model_type)
        train_y, test_y = split_subjects(data_y,data_index,sub_test_index,model_type=None)
        print(f'Test subjects for fold {fold}: {data_index[sub_test_index,1]}')
      else:
        train_x, test_x = split_data(data_input, Parts=fold_num, part_index=fold, shuffle=shuffle,model_type=model_type)
        train_y, test_y = split_data(data_y, Parts=fold_num, part_index=fold, shuffle=True,model_type=None)
      # Generating categorical data
      train_yc = to_categorical(train_y)
      test_yc = to_categorical(test_y)
      num_classes = train_yc.shape[1]


      # Building and validating  model
      model = build_model(train_x, train_yc, model_type=model_type,show_summary=show_summary, batch_norm=batch_norm ,apply_spectral=apply_spectral,dropout_rate=dropout_rate)    
      model.fit(train_x, train_yc, epochs=epochs, batch_size=batch_size, verbose=verbose,  validation_data=(test_x, test_yc))

      # fixing memory leak
      K.clear_session()
      tf.compat.v1.reset_default_graph() # TF graph isn't same as Keras graph
      gc.collect()


      metric_data.append(model.history.history)
      fold_time = time.time() - start_time
      time_data.append(fold_time)
      if align_to_subject:
        subject_data.append(data_index[sub_test_index,1])

      # saving partial results
      metric_results[model_type]=metric_data
      time_results[model_type]=time_data
      if align_to_subject:
        subject_results[model_type] = subject_data
        save_results(metric_results,time_results,dataset, Bands, subject_results=subject_results, model_type=model_type,partial=True)
      else:
        save_results(metric_results,time_results,dataset, Bands,subject_results=None, model_type=model_type, partial=True)




      print(f'Fold {fold} done in {timedelta(seconds=round(fold_time))} ')
    
    # saving final resutls for model
    metric_results[model_type]=metric_data
    time_results[model_type]=time_data
    if align_to_subject:
      subject_results[model_type] = subject_data
      save_results(metric_results,time_results, dataset, Bands,subject_results=subject_results,model_type=model_type)
    else:
      save_results(metric_results,time_results, dataset, Bands,subject_results=None,model_type=model_type)






  

  # Play an audio to alert for finishing
  if play_audio:
    from google.colab import output
    output.eval_js('new Audio("https://upload.wikimedia.org/wikipedia/commons/4/42/Bird_singing.ogg").play()')

  return model, metric_results, time_results,  (subject_results if align_to_subject else None)

########################## Function End #####################################

######################### Function Start ###################################
def predict_model(model,model_type, dataset, data_x, data_y,data_index,fold_num,Folds, align_to_subject=True, selected_subjects=False, selected_classes=False, shuffle=True):

  if fold_num > data_index.shape[0] and align_to_subject:
    raise ValueError('Number of folds should be less than number of subjects.')
  elif fold_num>data_x.shape[0]:
    raise ValueError('Number of folds should be less than number of trials.')

  if selected_classes:
    data_x, data_y, data_index = keep_classes(data_x, data_y, data_index, selected_classes)
  if selected_subjects:
    data_x, data_y, data_index = keep_subjects(data_x, data_y, data_index, selected_subjects)




  if model_type in ['CNN3D', 'TimeDist']: # these model requires 2D mapped data
    # Generating 2D mapped data
    pos_map = get_pos_map(dataset) # positions for 2D map conversion
    data_input = make_into_2d(data_x,pos_map)
  else:
    data_input = data_x

  if align_to_subject:# distributing subjects on folds in a fair way
    N = data_index.shape[0]
    sub_part = np.ones(fold_num)*np.floor(N/fold_num)
    sub_part[0:(N % fold_num)] +=1
    sub_part_index = np.append(0,np.cumsum(sub_part)).astype(int)

  for fold in Folds:
    print(f'Prediction accuracy for fold {fold}')
    # Splitting data 
    if align_to_subject:
      sub_test_index = np.arange(sub_part_index[fold],sub_part_index[fold+1])
      _, test_x = split_subjects(data_x,data_index,sub_test_index,model_type=model_type)
      _, test_y = split_subjects(data_y,data_index,sub_test_index,model_type=None)
    else:
      _, test_x = split_data(data_input, Parts=fold_num, part_index=fold, shuffle=shuffle,model_type=model_type)
      _, test_y = split_data(data_y, Parts=fold_num, part_index=fold, shuffle=True,model_type=None)


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
  if data_x.shape[ax] % N != 0: raise ValueError("Time samples in input data array must be multiple of N")
  if N==1: return data_x, data_y
  
  new_shape = list(data_x.shape)
  new_shape[ax-1] = data_x.shape[ax-1]*N
  new_shape[ax] = int(data_x.shape[ax]/N)
  data_x = np.reshape(data_x, new_shape)

  data_y = np.repeat(data_y,N)

  if data_index is not None:
    data_index_new=data_index.copy()
    data_index_new *= N
    return data_x, data_y, data_index
  else:
    return data_x, data_y
########################## Function End ######################################


######################### Function Start #####################################
def plot_results(datasets=['physionet', 'ttk', 'bcicomptIV2a'], metrics=None,ylim=[0,100],show=True,save=True):
  import glob
  from os import path
  from io import StringIO
  if save:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    if path.exists(f'{this.RESULT_FOLDER}/plots/') == False: os.mkdir(f'{this.RESULT_FOLDER}/plots/')
  if metrics is None:
    metrics=this.METRICS_TO_SAVE

  y_min = ylim[0]
  y_max = ylim[1]
  colors = ['red', 'blue','green','black','cyan',  'magenta' ,'brown','gray','teal']
  markers = ['s', '^', 'o', 'v','d','>','<','*','.']
  linestyles = ['solid','dotted']

  for dataset in datasets:
    for metric in metrics:
      files = sorted(glob.glob(f'{this.RESULT_FOLDER}/{dataset}*{metric}*.txt'))
      if len(files) == 0: continue
      if "val_" not in metric:
        files = [item for item in files if "_val_" not in item]
      sub_files = sorted(glob.glob(f'{this.RESULT_FOLDER}/{dataset}*subjects*.txt'))
 

      DataSZ = 0
      for index, file in enumerate(files):
        # finding subject count for each fold
        try:
          with open(sub_files[index],'r') as f: txt = f.read()
          txt=txt.replace('[', '').replace(']', '').split(',')
          sub_count = []
          for split in txt: sub_count.append(np.loadtxt(StringIO(split),dtype=int).size)
        except:
          sub_count = None
          
        data = 100 * np.genfromtxt(file, delimiter=',')
        if DataSZ == 0: DataSZ = data.shape[-1]

        if len(data.shape)==1:
          mean = data
        else:
          if sub_count == None:
            mean = np.mean(data,axis=0)
          else:
            mean = np.sum(np.transpose(sub_count * np.transpose(data)),axis=0)/np.sum(sub_count)
        
        Name = os.path.basename(file).split('.')[0]
        Name = Name.split('=')[1]
        #Name = Name.split('=')[1] + ('_MF' if Name.split('=')[2] == 'bandson' else '')
        plt.plot(mean,color=colors[index],linestyle = linestyles[0 if index <len(colors) else 1],linewidth=.7,marker=markers[index],markersize=5, markevery=list(np.arange(int(DataSZ/5),DataSZ,int(DataSZ/5))), label=Name)

      metric_name = metric.replace("cohen_", "")
      metric_name = metric_name if "val_" not in metric_name else f'validation {metric_name.split("_")[1]}'

      plt.xlabel('Epochs')
      plt.ylabel(f'Average {metric_name} %')
      plt.grid()
      plt.ylim(y_min,y_max)
      plt.yticks(np.arange(y_min,y_max+1,5))

      from matplotlib.ticker import MultipleLocator
      plt.gca().yaxis.set_minor_locator(MultipleLocator(4))
      plt.gca().yaxis.grid(True, which='minor',linestyle='dotted')
      plt.minorticks_on()
      
      dataset_name = {'bcicomptIV2a':'BCI Competition IV-2a', 'physionet':'Physionet','ttk':'MTA-TTK'}
      plt.title(f'Average {metric_name} for {dataset_name[dataset]} dataset.')
      plt.legend(fontsize=8)
      if save: plt.savefig(f'{this.RESULT_FOLDER}/plots/{dataset}={Name}={metric}={now}.pdf')
      if show: plt.show()
      plt.close()
  return
########################## Function End ######################################

######################### Function Start #####################################
def average_results(datasets=['physionet', 'ttk', 'bcicomptIV2a'], metrics=None,epochs=50, show=True,save=True):
  import glob
  from os import path
  from io import StringIO
  if metrics is None:
    metrics=this.METRICS_TO_SAVE

  if save:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_h = open(f'{this.RESULT_FOLDER}/average={now}.txt', "w")

  for dataset in datasets:
    files = glob.glob(f'{this.RESULT_FOLDER}/{dataset}*.txt')

    if len(files)==0:
      continue

    dataset_line = f'===========================\n           {dataset}          \n****************************'
    if show:
      print(dataset_line)
    if save:
      file_h.write(dataset_line+'\n')


    for metric in metrics:
      metric_name = metric.replace("cohen_", "")
      metric_name = metric_name.capitalize() if "val_" not in metric_name else f'Validation {metric_name.split("_")[1].capitalize()}'

      heading = f'-------- {metric_name}-----------'
      if save:
        file_h.write(heading+'\n')
      if show:
        print(heading)
      files = sorted(glob.glob(f'{this.RESULT_FOLDER}/{dataset}*{metric}*.txt'))
      if "val_" not in metric:
        files = [item for item in files if "_val_" not in item]
      if len(files) == 0: continue
      sub_files = sorted(glob.glob(f'{this.RESULT_FOLDER}/{dataset}*subjects*.txt'))


      DataSZ = 0
      for index, file in enumerate(files):
        # finding subject count for each fold
        try:
          with open(sub_files[index],'r') as f: txt = f.read()
          txt=txt.replace('[', '').replace(']', '').split(',')
          sub_count = []
          for split in txt: sub_count.append(np.loadtxt(StringIO(split),dtype=int).size)
        except:
          sub_count = None
        
        data = 100 * np.genfromtxt(file, delimiter=',')
        if DataSZ == 0: DataSZ = data.shape[-1]

        if len(data.shape)==1:
          mean =   data
        else:
          if sub_count == None:
            mean = np.mean(data, axis=0)
          else:
            mean = np.sum(np.transpose(sub_count * np.transpose(data)),axis=0)/np.sum(sub_count)

        Name = os.path.basename(file).split('.')[0]
        Name = Name.split('=')[1]
        #Name = Name.split('_')[1] + ('_MF' if Name.split('_')[2] == 'bandson' else '')

        line = f'{Name:15}Mean({DataSZ-epochs}-{DataSZ}) = {np.mean(mean[DataSZ-epochs:DataSZ]):0.1f}'
        if save:
          file_h.write(line+'\n')
        if show:
          print(line)
  if save:
    file_h.close()

  return
########################## Function End ######################################



######################### Function Start #####################################
def validation_time(datasets=['physionet', 'ttk', 'bcicomptIV2a'], show=True, save=True):
  import glob
  from os import path
  from io import StringIO

  def time(s):
    s = int(s)
    h = s//3600
    s = s % 3600
    m = s // 60
    s = s % 60
    return f'{h:02n}:{m:02n}:{s:02n}'
  if save:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_h = open(f'{this.RESULT_FOLDER}/duration={now}.txt', "w")

  for dataset in datasets:
    files = glob.glob(f'{this.RESULT_FOLDER}/{dataset}*.txt')

    if len(files)==0:
      continue

    dataset_line = f'===========================\n           {dataset}          \n****************************'
    if show:
      print(dataset_line)
    if save:
      file_h.write(dataset_line+'\n')

    files = sorted(glob.glob(f'{this.RESULT_FOLDER}/{dataset}*time*.txt'))
    files = [item for item in files if "_validation_" not in item]
    if len(files) == 0: continue
    for file in files:
      Name = os.path.basename(file).split('.')[0]
      Name = Name.split('=')[1]
      #Name = Name.split('_')[1] + ('_MF' if Name.split('_')[2] == 'bandson' else '')

      data = np.genfromtxt(file, delimiter=',')
      line = f'{Name:15}   {time(np.round(np.sum(data)))}'
      if save:
        file_h.write(line+'\n')
      if show:
        print(line)
  if save:
    file_h.close()

  return
########################## Function End ######################################





######################### Function Start #####################################
def save_results(metric_results,time_results,dataset,Bands,subject_results=None,model_type=None, partial=False):
  Band_status= 'bandson' if Bands else 'bandsoff' 
  now = datetime.now().strftime("%Y%m%d_%H%M%S") if not partial else "partial"
  models=[model_type] if model_type else metric_results.keys()


  # remove partial result files 
  if not partial:
    command = f'rm {this.RESULT_FOLDER}/{dataset}={model_type}={Band_status}=*=partial.txt'
    os.system(command)


  # saving metrics
  for metric in this.METRICS_TO_SAVE:
    for model_type in models:
      metric_data=metric_results[model_type]
      file_h = open(f'{this.RESULT_FOLDER}/{dataset}={model_type}={Band_status}={metric}={now}.txt', "w")
      writer = csv.writer(file_h)
      for split in metric_data:
        d = split[metric]
        writer.writerow([f'{x:0.4f}' for x in d])
      file_h.close()

  for model_type in models:
    time_data=time_results[model_type]
    file_h = open(f'{this.RESULT_FOLDER}/{dataset}={model_type}={Band_status}=time={now}.txt', "w")
    writer = csv.writer(file_h)
    writer.writerow([f'{x:0.2f}' for x in time_data])
    file_h.close()

  if subject_results==None:
    print('Data saved')
    return


  for model_type in models:
    subject_data=subject_results[model_type]
    file_h = open(f'{this.RESULT_FOLDER}/{dataset}={model_type}={Band_status}=subjects={now}.txt', "w")
    writer = csv.writer(file_h)
    writer.writerow(subject_data)
    file_h.close()


  if not partial: print('Data saved')
  return
########################## Function End ######################################


######################### Function Start #####################################
def dct_1d(data_x):
  from scipy import fft

  # find 1 dimentional dct for x_data
  data_x_out = fft.dct(data_x, axis=1)

  return data_x_out
########################## Function End ######################################

######################### Function Start #####################################
def fft_1d(data_x):
    # find 1 dimentional fft for x_data
  data_x_out = np.abs(np.fft.fft(data_x, axis=1))

  return data_x_out
########################## Function End ######################################
