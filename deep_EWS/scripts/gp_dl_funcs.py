# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 13:31:10 2018
@author: ball4624
"""
import numpy as np
import tensorflow as tf
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
rn.seed(12345)
# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras.models import Model
from keras import backend as K
from keras.utils.vis_utils import plot_model 
from keras.models import load_model
from keras.layers import merge, LSTM, Dense, SimpleRNN, Masking, Bidirectional, Dropout, concatenate, Embedding, TimeDistributed, multiply, add, dot, Conv2D
from attention_utils import get_activations, get_data_recurrent

import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import time
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import Adam, Adagrad, SGD
from keras import regularizers, callbacks
from keras.utils.vis_utils import plot_model 
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score

import os
import matplotlib.pyplot as plt 

from keras.layers.core import *
from keras.models import *
from custom_recurrents import AttentionDecoder
from keras.activations import tanh, softmax
from keras.layers import Lambda, Flatten
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
import sklearn.metrics as metrics


# Function so split observation set to training and test set
def split_dataset(t1, t2, t3, t4, t5, t6, obs):
    train_set = obs.loc[(obs.admittime>=t1)&(obs.admittime<=t2)]
    val_set = obs.loc[(obs.admittime>=t3)&(obs.admittime<=t4)]
    test_set = obs.loc[(obs.admittime>=t5)&(obs.admittime<=t6)]
    print ('TEST SET STATS')
    print('Number of patients:  ' + str(len(test_set.drop_duplicates(subset = 'hadm_id'))))
    print('Number of observations: ', len(test_set))
    print ('VAL SET STATS')
    print('Number of patients:  ' + str(len(val_set.drop_duplicates(subset = 'hadm_id'))))
    print('Number of observations: ', len(val_set))
    print ('TRAINING SET STATS')
    print('Number of patients:  ' + str(len(train_set.drop_duplicates(subset = 'hadm_id'))))
    print('Number of observations: ', len(train_set))
    len_test = len(test_set)
    len_train = len(train_set)
    len_val = len(val_set)
    print('Training set is ' + str(round((len_train/(len(obs)))*100, 2)) + ' % of overall observation set')
    print('Test set is ' + str(round((len_test/(len(obs)))*100, 2)) + ' % of overall observation set')
    print('Val set is ' + str(round((len_val/(len(obs)))*100, 2)) + ' % of overall observation set')
    return train_set, val_set, test_set


# drops observations recorded after first event
def drop_first_event(train_set, val_set, test_set):
    test_set= test_set.drop(test_set[test_set.hrs_to_firstevent < 0].index)
    train_set= train_set.drop(train_set[train_set.hrs_to_firstevent < 0].index)
    val_set= val_set.drop(val_set[val_set.hrs_to_firstevent < 0].index)
    test_set= test_set.drop(test_set[test_set.hrs_to_discharge < 0].index) # this would only happen when the discharge date is wrong
    train_set= train_set.drop(train_set[train_set.hrs_to_discharge < 0].index)
    val_set= val_set.drop(val_set[val_set.hrs_to_discharge < 0].index)
    print('\nTest Set:')
    print('Number of patients excluding obs after first event:  ' + str(len(test_set.drop_duplicates(subset = 'hadm_id'))))
    print('Number of observations: ', len(test_set))
    print('\nVal Set:')
    print('Number of patients excluding obs after first event:  ' + str(len(val_set.drop_duplicates(subset = 'hadm_id'))))
    print('Number of observations: ', len(val_set))
    print('\nTrain Set:')
    print('Number of patients excluding obs after first event:  ' + str(len(train_set.drop_duplicates(subset = 'hadm_id'))))
    print('Number of observations: ', len(train_set))
    return train_set, val_set, test_set


#%% Plot the priors 
def plot_lognormal(mu, sigma, c, l, xlim, ylim):
    s = np.random.lognormal(mu, sigma, 1000)
    x = np.linspace(0, xlim, 1000)
    pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))/ (x * sigma * np.sqrt(2 * np.pi)))
    plt.plot(x, pdf, linewidth=1, color=c, label=l)
    plt.axis('tight')
    plt.ylim(0,ylim)
    #plt.show()
    
    
    #%%
def df_hyperparams_vital(vital, hyperparams_results, patient_ids, label):
    gp_params= pd.DataFrame(hyperparams_results[vital])
    gp_params['hadm_id'] = patient_ids
    gp_params['label'] = np.full((1,len(patient_ids)), label)[0]
    df=gp_params.sort_values(['obj_func'])
    return df

def classify_hyperparams_svm(vital):
    df_HR_abn = df_hyperparams_vital(vital, hyperparams_abn, chosen_abnormals, 1)
    df_HR_norm = df_hyperparams_vital(vital, hyperparams_norm, chosen_normals, 0)
    all_HR = pd.concat([df_HR_abn, df_HR_norm])
    all_HR = all_HR.sample(frac=1).reset_index(drop=True)
    test = all_HR[1000:1204]
    train= all_HR[0:999]
    
    classify_param = 'rbf_lengthscales'
    clf_svm = svm.SVC(kernel='rbf', probability=True, random_state=137)
    clf_svm.fit(train[classify_param].reshape((999,1)), train.label)
    # Test on test set 
    Y_pred = clf_svm.predict(test[classify_param].reshape((204,1)))
    print('Accuracy on test set= ' + str(accuracy_score(test.label, Y_pred)))
    
    
#%%
def flt_data(X, Y):
    X_flt=[]
    Y_flt=[]
    for i in range(0, len(X)):
        for n in range(0, len(X[i])):
            X_flt.append(X[i][n])
            Y_flt.append(Y[i])
    return X_flt, Y_flt


#%%
    
def shuffle(r, X, Y):
    c = list(zip(X, Y))
    rn.shuffle(c, lambda:r)
    X, Y = zip(*c)
    return X, Y


#%%
def undersample_normals(obs, frac, n):
# this function includes all abnormals and undersamples normals
# frac = frac of abnormals in the dataset
# n is the number of patients in total 
    abnormals = obs.loc[obs.label>0]
    normals = obs.loc[obs.label ==0]
    len_abnormals = int(n*frac)
    len_normals = int(((1-frac)/frac)*len_abnormals)
    sampled_abnormals = abnormals.sample(n=len_abnormals, random_state=5)
    sampled_normals = normals.sample(n=len_normals, random_state=5)    
    sampled_hadmid = pd.concat([sampled_abnormals, sampled_normals])
    return sampled_hadmid

def sensitivity(y_true, y_pred):
    TP = np.logical_and((y_true==1), (y_pred.flatten()==1) )
    TP= sum(TP)
    FN = np.logical_and((y_true==1), (y_pred.flatten()==0) )
    FN = sum(FN)
    return TP/(TP+FN)
    
def specificity(y_true, y_pred):
    TN= np.logical_and((y_true==0), (y_pred.flatten()==0) )
    TN = sum(TN)
    FP = np.logical_and((y_true==0), (y_pred.flatten()==1) )
    FP = sum(FP)
    return TN/(TN+FP)




def spec_at_sens(sens, y_pred_prob, y_true, r):
    thresh = np.arange(0, 20, r)
    adj_thresh = 0
    t=0
    while t<len(thresh):
        y_pred = y_pred_prob >= np.round(thresh[t]/20,3)
        sn = sensitivity(y_true, y_pred)
        print(sn)
        if np.round(sn, 2) == sens:
            adj_thresh = thresh[t]
            break
        elif np.round(sn,2) < sens:
            adj_thresh = thresh[t]
            thresh = np.arange(0, adj_thresh, r)
            t=0
        else:
            t=t+1
        
    y_pred = y_pred_prob >= np.round(adj_thresh/20, 3)
    print(adj_thresh)
    return specificity(y_true, y_pred)





#%% This function returns a dataframe of performance metrics for models investigated
# models are passed as a dictionary 
def performance(models, hr, metrics , fit_models, test_label,   N=0, v_len=1, ts=12, m_feat='HR_mean'):
    # N is binary is to assess performance at n hours prior to event [-1, -2, -3] --> [2, 4, 6] hrs
    performance_metrics = pd.DataFrame(columns=['model'] + list(metrics.keys()))
    Y_pred_td = {}
    Y_test_td={}

    for key in models: 
        print(key)
        if (key == 'NEWS')|(key=='ASEWS')|(key=='LR'):
            test_set = models[key][test_label]
            # ews_set = reorder_testset(ews_set)
            Y_pred = models[key]['Y']['pred'].values
            Y_test = models[key]['Y']['true'].values
        
        else: # if deep learning model 
            #single_y = models[key]['y_format']
            #label = models[key]['label']
            #Y_test = output_format_2(models[key][test_label], single_y, label)
            mean_features = models[key]['mean_features']
            if models[key]['variance'] == 0:
                var_features =0
            else:
                var_features = models[key]['var_features'] 
            test_set = models[key][test_label]
            # load weights
            K.clear_session()
            m = load_model(fit_models[key])
            if 'MLP' in key:
                ts=1
            Y_pred, Y_test, len_seq, hadm_id, hrs_from_adm, _ = test_batches(test_set, mean_features, m, var_features, models[key]['variance'],
                                                                             models[key]['single_attention_vital'], models[key]['features_descrip'], v_len=v_len, m_feat=m_feat, ts=ts)
        
        
        Y_pred_td[key] = Y_pred
        Y_test_td[key] = Y_test
        Y_pred_proc = np.array(Y_pred).flatten()
        
        print(len(Y_pred))
        print(len(Y_test))
        
        if key =='NEWS': 
            thresh = [0.25]
        else: 
            thresh = [0.5]
        # np.linspace(0.1, 0.25, num=50, endpoint=True)
        for i in range(0, len(thresh)):
            Y_pred_bin = Y_pred_proc>= thresh[i]
            scores=[key]
            for m in metrics:
                metric = m
                func = metrics[m]
                if (metric == 'ACC') | (metric== 'F1 score') | (metric=='SENS') | (metric=='SPEC') | (metric=='PPV'):
                    scores.append(round(func(Y_test, Y_pred_bin),3))
                else:
                    scores.append(round(func(Y_test, Y_pred_proc),3))
            performance_metrics=performance_metrics.append(pd.DataFrame([scores], columns=['model']+list(metrics.keys())))
    len_seq = 0
    hadm_id =0
    hrs_from_adm=0
    return performance_metrics, Y_pred_td, Y_test_td, len_seq, hadm_id, hrs_from_adm
    

    

#%% predictions for encoder-decoder model
OUTPUT_LENGTH=12
def generate(i,m):
    encoder_input = i
    decoder_input = np.zeros(shape=(1,12, 1))
    decoder_input[:,0] = 5
    for i in range(1, OUTPUT_LENGTH):
        output = m.predict([encoder_input, decoder_input], batch_size=1).argmax(axis=2)
        decoder_input[:,i] = output[:,i]
    output = np.zeros(shape=(1,12,1))
    output[:,:-1] = decoder_input[:,1:]
    output[:,-1] =  m.predict([encoder_input, decoder_input], batch_size=1).argmax(axis=2)[:,-1]
    return output

def to_katakana(i,m):
    decoder_output = generate(i,m)
    return decoder_output.flatten()


def predict_enc_dec(x,m):
    t_pred=[]
    for t in x:
        t = t.reshape(1,12,7)
        t_out = to_katakana(t,m)
        if len(t_pred)==0:
            t_pred =[t_out]
        else:
            t_pred = np.append(t_pred, [t_out], axis=0)
    return t_pred


#%% Reformat test set based on batches
def test_batches(test_set, mean_features, m, var_features, variance=0, single_attention_vital=0, d='', ts=12, v_len=1, m_feat = 'HR_mean'):
    Y_pred=[]
    Y_test=[]
    len_seq = []
    hadm_id = []
    hrs_from_adm = []
    codes = []
    if v_len ==1:
        maxx = ts+1
    else:
        maxx = 2
    print(ts)
    for n in range(1,13): # this doesn't seem right (it gets results for validation set etc 3 times)
        print(n)
        if v_len==1:
            sub_test_set = test_set.loc[test_set.len_sequence==n]
            unique_sub_test_set = sub_test_set.drop_duplicates(subset='code', keep='last')
        else:
            sub_test_set = test_set
            unique_sub_test_set = sub_test_set.drop_duplicates(subset='code', keep='last')
        
        
        if len(Y_test)==0:
            Y_test = format_labels(sub_test_set,-5, m_feat).values
            if v_len == 1:
                len_seq = unique_sub_test_set.len_sequence.values
            hadm_id = unique_sub_test_set.hadm_id.values
            hrs_from_adm = unique_sub_test_set.hrs_from_adm.values
            codes = unique_sub_test_set.code.values
        else:
            Y_test = np.append(Y_test, format_labels(sub_test_set,-5, m_feat).values)
            if v_len==1:
                len_seq = np.append(len_seq, unique_sub_test_set.len_sequence.values)
            hadm_id = np.append(hadm_id, unique_sub_test_set.hadm_id.values)
            hrs_from_adm =np.append(hrs_from_adm, unique_sub_test_set.hrs_from_adm.values)
            codes = np.append(codes, unique_sub_test_set.code.values)
        if (single_attention_vital ==1)&(variance==0):
            input_feat = []
            for feat in mean_features:
                input_feat = input_feat +  [np.array(reshape_features(sub_test_set, [feat], ts))]
              
            
            mean_input=input_feat
            y= m.predict(mean_input)
        elif (single_attention_vital==0)&(variance==0):
            print('here')
            mean_input = reshape_features(sub_test_set, mean_features, ts)
            y= m.predict(mean_input)
        elif (single_attention_vital==0)&(variance==1):
            mean_input = reshape_features(sub_test_set, mean_features, ts)
            var_input = reshape_features(sub_test_set, var_features, ts)
            y=m.predict([mean_input, var_input])    

        elif (single_attention_vital==1)&(variance==1):
            i1 = reshape_features(sub_test_set, ['HR'+d], ts)
            i2 = reshape_features(sub_test_set, ['RR'+d], ts)
            i3 = reshape_features(sub_test_set, ['SBP'+d], ts)
            i4 = reshape_features(sub_test_set, ['SPO2'+d], ts)
            i5 = reshape_features(sub_test_set, ['TEMP'+d], ts)
            i6 = reshape_features(sub_test_set, ['avpu'+d], ts)
            i7 = reshape_features(sub_test_set, ['supplemental_oxygen'+d], ts)            
            v='_var'
            v1 = reshape_features(sub_test_set, ['HR'+v], ts)
            v2 = reshape_features(sub_test_set, ['RR'+v], ts)
            v3 = reshape_features(sub_test_set, ['SBP'+v], ts)
            v4 = reshape_features(sub_test_set, ['SPO2'+v], ts)
            v5 = reshape_features(sub_test_set, ['TEMP'+v], ts)
            y=m.predict([i1,i2, i3, i4, i5, i6, i7, v1, v2, v3, v4, v5]) 
            
        elif single_attention_vital ==2:
            v='_var'
            i1 = reshape_features(sub_test_set, ['HR'+d, 'HR'+v], ts)
            i2 = reshape_features(sub_test_set, ['RR'+d, 'RR'+v], ts)
            i3 = reshape_features(sub_test_set, ['SBP'+d, 'SBP'+v], ts)
            i4 = reshape_features(sub_test_set, ['SPO2'+d, 'SPO2'+v], ts)
            i5 = reshape_features(sub_test_set, ['TEMP'+d, 'TEMP'+v], ts)
            i6 = reshape_features(sub_test_set, ['avpu'+d], ts)
            i7 = reshape_features(sub_test_set, ['supplemental_oxygen'+d], ts)            
            
           
            y=m.predict([i1,i2, i3, i4, i5, i6, i7]) 
               
        elif variance ==2:
            y=predict_enc_dec(mean_input,m)
        if len(Y_pred)==0:
            Y_pred = [y[i][-1] for i in range(0, len(y))]
            #Y_pred = y
        else:
            Y_pred= np.append(Y_pred, [y[i][-1] for i in range(0, len(y))])
    return Y_pred, Y_test, len_seq, hadm_id, hrs_from_adm, codes
       
def reorder_testset(dataset):
    new_set = pd.DataFrame(columns = list(dataset))
    for n in range(1,13):
        subset = dataset.loc[dataset.len_sequence==n]
        new_set=pd.concat([new_set, subset])
    return new_set

#%% Plot histories
def plot_history(history, figs_path, epochs):
    # list all data in history
    for key in history:
        # summarize history for accuracy
        plt.plot(history[key]['binary_accuracy'], 'r--', label='Training accuracy')
        plt.plot(history[key]['val_binary_accuracy'], 'b--', label='Validation accuracy')
        plt.title(key+' Training History Accuracy ')
        plt.ylabel('Accuracy ')
        plt.xlabel('Epoch')
        plt.legend( loc='upper left')
        plt.axis((0,epochs,0,1))
        plt.savefig(figs_path+key+'model_accuracy_history.png')
        plt.show()
        
        plt.plot(history[key]['loss'], 'r-', label = 'Training loss')
        plt.plot(history[key]['val_loss'], 'b-', label = 'Validaion loss')
        plt.title(key+' Training History Loss ')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend( loc='upper left')
        plt.axis((0,epochs,0,1))
        plt.savefig(figs_path+key+'model_loss_history.png')
        plt.show()
        
        
        
        
#%% Compute performance per hour
def performance_per_hour(figs_path, all_models, hrs, metrics, Y_test, fit_models, test_set):
    performance_per_hour={}
    Y_pred_td_per_hour={}
    for i in range(0, len(hrs)):
        print(i)
        pm, y_pred_td_per_hour = performance(all_models, int(hrs[i]), metrics, fit_models, test_set, 1 )
        performance_per_hour[i]=pm
        Y_pred_td_per_hour[i] = y_pred_td_per_hour
    return performance_per_hour, Y_pred_td_per_hour

#%% Plot performance per hour      
def plot_performance_per_hour(performance_per_hour, metrics, figs_path):
    for m in metrics:
        aurocs_per_hour = []
        for key in performance_per_hour:
          aurocs_per_hour.append(performance_per_hour[key][m].values)  
        aurocs_per_hour = np.transpose(aurocs_per_hour)
    
        colors = ['g', 'b', 'r', 'y']
        labels=performance_per_hour[0].model.values
        fig, ax= plt.subplots()
        for n in range(0, len(aurocs_per_hour)): 
            plt.plot(np.arange(0, 18, step=1), aurocs_per_hour[n], colors[n]+'o', label=labels[n])
            plt.axis((0,17,0,1.1))
        #temp_x = reversed(np.arange(0, 40, step = 4))

        plt.title(m)
        ax.set_xticklabels(np.arange(0, 40, step = 4)[::-1])
        #plt.xticks(np.arange(36,0, step=-2))
        plt.xlabel('Hours to Event')
        plt.legend()
        plt.savefig(figs_path+m+'_at_every_hour.png', dpi=300)
        plt.show()    
    #%%
    
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False

def attention_3d_block(inputs, num):
    # inputs.shape = (batch_size, time_steps, input_dim)
    # num is used to label the attention_3d_blocks used in your model 
    input_dim = int(inputs.shape[2])
    #TIME_STEPS = int(inputs.shape[1])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, 12))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(12, activation='relu')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec_'+str(num))(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul_'+str(num), mode='mul')
    return output_attention_mul


# This attention block performs self-attention, since we are not comparing our source hidden states
    # to any target hidden states
def attention_3d_block_2(inputs_1, num):
    # num is used to label the attention_3d_blocks used in your model 
    input_dim = int(inputs_1.shape[2])
    
    # 1. Compute eij i.e. scoring function (aka similarity function)
    eij = Permute((2, 1))(inputs_1)
    eij = Reshape((input_dim, 12))(eij) # this line is not useful. It's just to know which dimension is what.
    eij = Dense(12, activation='tanh')(eij)
    
    if SINGLE_ATTENTION_VECTOR:
        eij = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(eij)
        eij= RepeatVector(input_dim)(eij)
        
    # 2. Compute attention probabilities by normalizing 
    a_probs = Lambda(lambda x: K.exp(x))(eij)
    sum_a_probs = Lambda(lambda x: 1/ K.cast(K.sum(x, axis=1, keepdims=True) + K.epsilon(), K.floatx()))(a_probs)
    a_probs = multiply([a_probs, sum_a_probs], name='attention_vec_'+str(num))
    # a_probs = Permute((2, 1), name='attention_vec_'+str(num))(a)
    
    # 3. Compute context vector c = sum(attention weights*source hidden states)
    output_attention_mul = merge([inputs_1, a_probs], name='attention_mul_'+str(num), mode='mul')
    result = Lambda(lambda x: K.sum(x, axis=1))(output_attention_mul)    
    
    # 4. Missing step: compute the attention vector a = f(h_t) 
    return result

def attention_3d_block_3(inputs_1, inputs_2, num, num_2):
    # inputs.shape = (batch_size, time_steps, input_dim)
    # num is used to label the attention_3d_blocks used in your model 
    input_dim = int(inputs_1.shape[2])
    #TIME_STEPS = int(inputs.shape[1])
    # Compute eij
    eij = Permute((2, 1))(inputs_1)
    eij = Dense(12, activation='tanh')(eij)
    
    # Compute attention probabilities by normalizing 
    a_probs = Lambda(lambda x: K.exp(x))(eij)
    sum_a_probs = Lambda(lambda x: 1/ K.cast(K.sum(x, axis=1, keepdims=True) + K.epsilon(), K.floatx()))(a_probs)
    
    a_probs = multiply([a_probs, sum_a_probs], name='attention_vec_'+str(num))
    
    output_attention_mul = merge([inputs_1, a_probs], name='attention_mul_'+str(num), mode='mul')
    output_attention_mul_2 = merge([inputs_2, a_probs], name='attention_mul_'+str(num_2), mode='mul')

    result_1 = Lambda(lambda x: K.sum(x, axis=1))(output_attention_mul)  
    result_2 = Lambda(lambda x: K.sum(x, axis=1))(output_attention_mul_2)  
    result = add([result_1, result_2])
    return result


# amending the attention equations (previously self attention and similarity function is incorrect)
def attention_3d_block_4(inputs_1, num):
    # num is used to label the attention_3d_blocks used in your model 
    
    # 1. Compute eij i.e. scoring function (aka similarity function)        
    # Type A: feed forward neural network 
    v1 = Dense(12)(inputs_1)
    v2 = Dense(12)(inputs_1)
    sum_v = add([v1,v2])
    sum_v = Activation('tanh')(sum_v)
    e = Dense(12)(sum_v)
    
    # Type B: Self-attention of hidden states
    #e = dot([inputs_1, inputs_1], axes =[2,2])

    # 2. Compute attention probabilities by normalizing 
    a_probs = Activation('softmax')(e)
     
    # 3. Compute context vector c = sum(attention weights*source hidden states)
    context = dot([a_probs, inputs_1 ], axes = [2,1])
    
    #merge([inputs_1, a_probs], name='attention_mul_'+str(num), mode='mul')
    #result = Lambda(lambda x: K.sum(x, axis=1))(output_attention_mul)    
    
    return context


def attention_3d_block_6(inputs_1, num):
    # num is used to label the attention_3d_blocks used in your model 
    
    # 1. Compute eij i.e. scoring function (aka similarity function)        
    # Type A: feed forward neural network 
    '''v1 = Dense(12)(inputs_1)
    v2 = Dense(12)(inputs_1)
    sum_v = add([v1,v2])
    sum_v = Activation('relu')(sum_v)
    e = Dense(12)(sum_v)'''
    
    # Type B: Self-attention of hidden states
    e = dot([inputs_1, inputs_1], axes =[2,2], normalize=True)

    # 2. Compute attention probabilities by normalizing 
    a_probs = Activation('softmax')(e)
     
    # 3. Compute context vector c = sum(attention weights*source hidden states)
    context = dot([a_probs, inputs_1 ], axes = [2,1])
    
    #merge([inputs_1, a_probs], name='attention_mul_'+str(num), mode='mul')
    #result = Lambda(lambda x: K.sum(x, axis=1))(output_attention_mul)    
    
    return context


def attention_3d_block_7(inputs_1, num):
    # num is used to label the attention_3d_blocks used in your model 
    
    # 1. Compute eij i.e. scoring function (aka similarity function)        
    # Type A: feed forward neural network 
    v1 = Dense(10, use_bias=True)(inputs_1)
    v1_tanh = Activation('relu')(v1)
    e = Dense(1)(v1_tanh)
    e_exp = Lambda(lambda x: K.exp(x))(e)
    sum_a_probs = Lambda(lambda x: 1/ K.cast(K.sum(x, axis=1, keepdims=True) + K.epsilon(), K.floatx()))(e_exp)
    a_probs = multiply([e_exp, sum_a_probs], name='attention_vec_'+str(num))
    
    #v2 = Dense(12)(inputs_1)
    #sum_v = add([v1,v2])
    #sum_v = Activation('relu')(sum_v)
    #e = Dense(12)(sum_v)
    
    # Type B: Self-attention of hidden states
    #e = dot([inputs_1, inputs_1], axes =[2,2], normalize=True)

    # 2. Compute attention probabilities by normalizing 
    #a_probs = Activation('softmax')(e)
     
    # 3. Compute context vector c = sum(attention weights*source hidden states)
    context = multiply([inputs_1,a_probs])
    context = Lambda(lambda x: K.sum(x, axis=1))(context) 
    
    #merge([inputs_1, a_probs], name='attention_mul_'+str(num), mode='mul')
    #result = Lambda(lambda x: K.sum(x, axis=1))(output_attention_mul)    
    
    return context


def ff_attention(i):
    v1 = Dense(12)(i)
    v2 = Dense(12)(i)
    sum_v = add([v1,v2])
    sum_v = Activation('tanh')(sum_v)
    e = Dense(12)(sum_v)
    return e

def logistic_regression(input_dim, timesteps):
    inputs = Input(batch_shape=(None, timesteps, input_dim))
    i = Flatten()(inputs)
    out = Dense(1, activation='sigmoid')(i)
    model = Model(input=[inputs], output=out)
    return model


# amending the attention equations (previously self attention and similarity function is incorrect)
def attention_3d_block_5(inputs_1, inputs_2):
    
    context_1 = attention_3d_block_7(inputs_1, 1)
    context_2 = attention_3d_block_7(inputs_2,2)

    context = add([context_1, context_2])
    
    #merge([inputs_1, a_probs], name='attention_mul_'+str(num), mode='mul')
    #result = Lambda(lambda x: K.sum(x, axis=1))(output_attention_mul)    
    
    return context


 


def lstm(INPUT_DIM, TIME_STEPS):
    inputs = Input(batch_shape=(None, TIME_STEPS, INPUT_DIM))
    lstm_units = TIME_STEPS
    #masking_layer = Masking(mask_value=-5)(inputs)
    lstm_out = LSTM(lstm_units, return_sequences=False,  input_shape=(12,7),kernel_regularizer=regularizers.l2(0.01), kernel_initializer='random_uniform')(inputs)
    output = Dense(1, activation='sigmoid', kernel_initializer='random_uniform')(lstm_out)
    model = Model(input=[inputs], output=output)
    return model


def bilstm(INPUT_DIM, TIME_STEPS):
    inputs = Input(batch_shape=(None, TIME_STEPS, INPUT_DIM))
    lstm_units = TIME_STEPS
    #masking_layer = Masking(mask_value=-5)(inputs)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=False,  input_shape=(12,7),kernel_regularizer=regularizers.l2(0.01), kernel_initializer='random_uniform'), 'ave')(inputs)
    output = Dense(1, activation='sigmoid', kernel_initializer='random_uniform')(lstm_out)
    model = Model(input=[inputs], output=output)
    return model


# model for labs 
def mlp():
    inputs  = Input(batch_shape=(None, 1, 7 ))
    l1= Dense(10)(inputs)
    l2=Dense(10)(l1)
    l3 =Dense(1, activation='tanh')(l2)
    output = Flatten()(l3)
    model = Model(input=[inputs], output=output)
    return model   

def lr():
    inputs  = Input(batch_shape=(None, 1, 7 ))
    l3 =Dense(1, activation='sigmoid')(inputs)
    output = Flatten()(l3)
    model = Model(input=[inputs], output=output)
    return model   




INPUT_LENGTH = 12
OUTPUT_LENGTH = 12
START_CHAR_CODE = 1


# attention applied to both aggregated of vital signs mean and variance and then summed up then decoder
# previously lstm_attention_variance_7
def lstm_attention_1(INPUT_DIM, TIME_STEPS):
    i = Input(shape=(TIME_STEPS,INPUT_DIM), dtype='float32')
    enc = Bidirectional(LSTM(5, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i)    
    dec = attention_3d_block_7(enc,1)    
    #dec = Permute((2,1))(dec)
    #dec = LSTM(12, return_sequences=True)(dec)
    output = Dense(5, activation='relu')(dec)
    #output = Flatten()(output)
    output = Dense(1, activation='sigmoid')(output)
    #output = TimeDistributed(Dense(1, activation='sigmoid'))(dec)
    model = Model(input=[i], output=output)
    model.summary()
    return model

# Attend to each vital sign separately 
# previously lstm_attention_variance_10
def lstm_attention_2(INPUT_DIM, TIME_STEPS):    
    i1 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc1 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i1)    
    dec1 = attention_3d_block_7(enc1,1)
    
    i2 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc2 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i2)    
    dec2 = attention_3d_block_7(enc2,2)
    
    
    i3 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc3 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i3)    
    dec3 = attention_3d_block_7(enc3,3)
    
    i4 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc4 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i4)    
    dec4 = attention_3d_block_7(enc4,4)
    
    
    i5 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc5 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i5)    
    dec5 = attention_3d_block_7(enc5,5)
    
    i6 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc6 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i6)    
    dec6 = attention_3d_block_7(enc6,6)
    
    i7 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc7 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i7)    
    dec7 = attention_3d_block_7(enc7,7)
    
    c_agg = add([dec1, dec2, dec3, dec4, dec5, dec6, dec7])
    
    #dec = LSTM(12, return_sequences=True)(c_agg)
    output = Dense(5, activation='relu')(c_agg)
    #output = Flatten()(output)
    output = Dense(1, activation='sigmoid')(output)

    #output = TimeDistributed(Dense(1, activation='sigmoid'))(dec)
    model = Model(input=[i1, i2, i3, i4, i5, i6, i7], output=output)
    model.summary()
    return model


#lstm_attention_2 for labs 
def lstm_attention_2_2(TS):    
    i1 = Input(shape=(TS,1), dtype='float32')
    enc1 = Bidirectional(LSTM(1, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i1)    
    dec1 = attention_3d_block_7(enc1,1)
    
    i2 = Input(shape=(TS,1), dtype='float32')
    enc2 = Bidirectional(LSTM(1, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i2)    
    dec2 = attention_3d_block_7(enc2,2)
    
    
    i3 = Input(shape=(TS,1), dtype='float32')
    enc3 = Bidirectional(LSTM(1, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i3)    
    dec3 = attention_3d_block_7(enc3,3)
    
    i4 = Input(shape=(TS,1), dtype='float32')
    enc4 = Bidirectional(LSTM(1, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i4)    
    dec4 = attention_3d_block_7(enc4,4)
    
    
    i5 = Input(shape=(TS,1), dtype='float32')
    enc5 = Bidirectional(LSTM(1, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i5)    
    dec5 = attention_3d_block_7(enc5,5)
    
    i6 = Input(shape=(TS,1), dtype='float32')
    enc6 = Bidirectional(LSTM(1, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i6)    
    dec6 = attention_3d_block_7(enc6,6)
    
    i7 = Input(shape=(TS,1), dtype='float32')
    enc7 = Bidirectional(LSTM(1, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i7)    
    dec7 = attention_3d_block_7(enc7,7)
    
    c_agg = add([dec1, dec2, dec3, dec4, dec5, dec6, dec7])
    
    #dec = LSTM(12, return_sequences=True)(c_agg)
    output = Dense(5, activation='relu')(c_agg)
    #output = Flatten()(output)
    output = Dense(1, activation='sigmoid')(output)

    #output = TimeDistributed(Dense(1, activation='sigmoid'))(dec)
    model = Model(input=[i1, i2, i3, i4, i5, i6, i7], output=output)
    model.summary()
    return model

#mlp with attention for labs 
def mlp_attention(TS):   
    i1 = Input(shape=(TS,1), dtype='float32')
    i1r = Reshape((1, TS))(i1)    
    enc1 = Dense(5)(i1r)
    
    i2 = Input(shape=(TS,1), dtype='float32')
    i2r = Reshape((1, TS))(i2) 
    enc2 = Dense(5)(i2r)
    
    i3 = Input(shape=(TS,1), dtype='float32')
    i3r = Reshape((1, TS))(i3) 
    enc3 = Dense(5)(i3r)
     
    i4 = Input(shape=(TS,1), dtype='float32')
    i4r = Reshape((1, TS))(i4) 
    enc4 = Dense(5)(i4r)
    
    i5 = Input(shape=(TS,1), dtype='float32')
    i5r = Reshape((1, TS))(i5) 
    enc5 = Dense(5)(i5r)
     
    i6 = Input(shape=(TS,1), dtype='float32')
    i6r = Reshape((1, TS))(i6) 
    enc6 = Dense(5)(i6r)
     
    i7 = Input(shape=(TS,1), dtype='float32')
    i7r = Reshape((1, TS))(i7) 
    enc7 = Dense(5)(i7r)
     
    enc_concat= concatenate([enc1, enc2, enc3, enc4, enc5, enc6, enc7], axis=1)
    dec = attention_3d_block_7(enc_concat,10)
    
    
    #dec = LSTM(12, return_sequences=True)(c_agg)
    output = Dense(5, activation='relu')(dec)
    #output = Flatten()(output)
    output = Dense(1, activation='sigmoid')(output)

    #output = TimeDistributed(Dense(1, activation='sigmoid'))(dec)
    model = Model(input=[i1, i2, i3, i4, i5, i6, i7], output=output)
    model.summary()
    return model





# previously lstm_attention_variance_3_2
def ua_lstm_attention_1(INPUT_DIM, TIME_STEPS):
    # uncertainty aware model
    mean_inputs = Input(batch_shape=(None, TIME_STEPS, 7))
    var_inputs =  Input(batch_shape=(None, TIME_STEPS, 5))
    lstm_units = TIME_STEPS
    lstm_out_mean = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(mean_inputs)    
    lstm_out_var = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(var_inputs)
    context = attention_3d_block_5(lstm_out_mean, lstm_out_var)
    #dec = LSTM(12, return_sequences=True)(context)
    dec = Dense(5, activation='relu')(context)
    output = Dense(1, activation='sigmoid')(dec)
    model = Model(input=[mean_inputs, var_inputs], output=output)
    model.summary()
    return model



# previously lstm_attention_variance_3_2
def ua_lstm_attention_2(INPUT_DIM, TIME_STEPS):
    i1 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc1 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i1)    
    dec1 = attention_3d_block_7(enc1,1)
    
    i2 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc2 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i2)    
    dec2 = attention_3d_block_7(enc2,2)
    
    
    i3 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc3 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i3)    
    dec3 = attention_3d_block_7(enc3,3)
    
    i4 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc4 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i4)    
    dec4 = attention_3d_block_7(enc4,4)
    
    i5 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc5 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i5)    
    dec5 = attention_3d_block_7(enc5,5)
    
    i6 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc6 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i6)    
    dec6 = attention_3d_block_7(enc6,6)
    
    i7 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc7 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i7)    
    dec7 = attention_3d_block_7(enc7,7)
    
    c_agg_1 = add([dec1, dec2, dec3, dec4, dec5, dec6, dec7])
    c_m = Dense(5)(c_agg_1)
    
    i12 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc12 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i12)    
    dec12 = attention_3d_block_7(enc12,8)
    
    i22= Input(shape=(TIME_STEPS,1), dtype='float32')
    enc22 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i22)    
    dec22 = attention_3d_block_7(enc22,9)
    
    
    i32 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc32 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i32)    
    dec32 = attention_3d_block_7(enc32,10)
    
    i42 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc42 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i42)    
    dec42 = attention_3d_block_7(enc42,11)
    
    
    i52 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc52 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i52)    
    dec52 = attention_3d_block_7(enc52,12)
    

    c_agg_2 = add([dec12, dec22, dec32, dec42, dec52])
    c_v = Dense(5)(c_agg_2)
    
    c = add([c_m, c_v])
    
    #dec = LSTM(12, return_sequences=True)(context)
    #dec = Dense(5, activation='relu', kernel_regularizer=regularizers.l2(0.01))(c)
    #dec2= BatchNormalization()(dec)
    dec = Dense(5, activation='relu')(c)
    dec2 = Dropout(0.2)(dec)
    output = Dense(1, activation='sigmoid')(dec2)
    model = Model(input=[i1, i2, i3, i4, i5, i6, i7, i12, i22, i32, i42, i52], output=output)
    model.summary()
    return model




'''def ua_lstm_attention_2(INPUT_DIM, TIME_STEPS):
    i1 = Input(shape=(12,1), dtype='float32')
    enc1 = LSTM(12, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform')(i1)    
    dec1 = attention_3d_block_7(enc1,1)
    
    i2 = Input(shape=(12,1), dtype='float32')
    enc2 = LSTM(12, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform')(i2)    
    dec2 = attention_3d_block_7(enc2,2)
    
    
    i3 = Input(shape=(12,1), dtype='float32')
    enc3 =LSTM(12, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform')(i3)    
    dec3 = attention_3d_block_7(enc3,3)
    
    i4 = Input(shape=(12,1), dtype='float32')
    enc4 = LSTM(12, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform')(i4)    
    dec4 = attention_3d_block_7(enc4,4)
    
    i5 = Input(shape=(12,1), dtype='float32')
    enc5 = LSTM(12, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform')(i5)    
    dec5 = attention_3d_block_7(enc5,5)
    
    i6 = Input(shape=(12,1), dtype='float32')
    enc6 = LSTM(12, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform')(i6)    
    dec6 = attention_3d_block_7(enc6,6)
    
    i7 = Input(shape=(12,1), dtype='float32')
    enc7 = LSTM(12, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform')(i7)    
    dec7 = attention_3d_block_7(enc7,7)
    
    c_agg_1 = add([dec1, dec2, dec3, dec4, dec5, dec6, dec7])
    c_m = Dense(5)(c_agg_1)
    
    i12 = Input(shape=(12,1), dtype='float32')
    enc12 =LSTM(12, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform')(i12)    
    dec12 = attention_3d_block_7(enc12,8)
    
    i22= Input(shape=(12,1), dtype='float32')
    enc22 = LSTM(12, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform')(i22)    
    dec22 = attention_3d_block_7(enc22,9)
    
    
    i32 = Input(shape=(12,1), dtype='float32')
    enc32 = LSTM(12, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform')(i32)    
    dec32 = attention_3d_block_7(enc32,10)
    
    i42 = Input(shape=(12,1), dtype='float32')
    enc42 =LSTM(12, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform')(i42)    
    dec42 = attention_3d_block_7(enc42,11)
    
    
    i52 = Input(shape=(12,1), dtype='float32')
    enc52 = LSTM(12, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform')(i52)    
    dec52 = attention_3d_block_7(enc52,12)
    
    c_agg_2 = add([dec12, dec22, dec32, dec42, dec52])
    c_v = Dense(5)(c_agg_2)
    
    c = add([c_m, c_v])
    
    #dec = LSTM(12, return_sequences=True)(context)
    #dec = Dense(5, activation='relu', kernel_regularizer=regularizers.l2(0.01))(c)
    #dec2= BatchNormalization()(dec)
    dec = Dense(5, activation='relu')(c)
    output = Dense(1, activation='sigmoid')(dec)
    model = Model(input=[i1, i2, i3, i4, i5, i6, i7, i12, i22, i32, i42, i52], output=output)
    model.summary()
    return model'''

def ua_lstm_attention_3(INPUT_DIM, TIME_STEPS):
    i1 = Input(shape=(TIME_STEPS,2), dtype='float32')
    enc1 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i1)    
    dec1 = attention_3d_block_7(enc1,1)
    
    i2 = Input(shape=(TIME_STEPS,2), dtype='float32')
    enc2 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i2)    
    dec2 = attention_3d_block_7(enc2,2)
    
    
    i3 = Input(shape=(TIME_STEPS,2), dtype='float32')
    enc3 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i3)    
    dec3 = attention_3d_block_7(enc3,3)
    
    i4 = Input(shape=(TIME_STEPS,2), dtype='float32')
    enc4 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i4)    
    dec4 = attention_3d_block_7(enc4,4)
    
    i5 = Input(shape=(TIME_STEPS,2), dtype='float32')
    enc5 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i5)    
    dec5 = attention_3d_block_7(enc5,5)
    
    i6 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc6 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i6)    
    dec6 = attention_3d_block_7(enc6,6)
    
    i7 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc7 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i7)    
    dec7 = attention_3d_block_7(enc7,7)
    
    c_agg_1 = add([dec1, dec2, dec3, dec4, dec5, dec6, dec7])
    c = Dense(5)(c_agg_1)
    
    
    #dec = LSTM(12, return_sequences=True)(context)
    #dec = Dense(5, activation='relu', kernel_regularizer=regularizers.l2(0.01))(c)
    #dec2= BatchNormalization()(dec)
    dec = Dense(5, activation='relu')(c)
    output = Dense(1, activation='sigmoid')(dec)
    model = Model(input=[i1, i2, i3, i4, i5, i6, i7], output=output)
    model.summary()
    return model



# old models  
def lstm_attention(INPUT_DIM, TIME_STEPS):
    #inputs = Input(batch_shape=(None, None, 7))
    inputs = Input(batch_shape=(None, 12, 7))
    lstm_units = 12
    # masking_layer = Masking(mask_value=-5)(inputs)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True,  kernel_regularizer=regularizers.l2(0.01), kernel_initializer='random_uniform'), 'ave')(inputs)
    attention_mul = attention_3d_block_2(lstm_out, 1)
    #output = TimeDistributed(Dense(1, activation='sigmoid', kernel_initializer='random_uniform'))(attention_mul)
    #attention_mul = Flatten()(attention_mul)
    output_1 = Dense(40, activation='relu')(attention_mul)

    output = Dense(1, activation='sigmoid', kernel_initializer='random_uniform')(output_1)
    model = Model(input=[inputs], output=output)
    return model


# attention applied to both mean and variance and then summed up
def lstm_attention_variance_3(INPUT_DIM, TIME_STEPS):
    # uncertainty aware model
    mean_inputs = Input(batch_shape=(None, 12, 7))
    var_inputs =  Input(batch_shape=(None, 12, 5))
    lstm_units = 12
    lstm_out_mean = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(mean_inputs)    
    lstm_out_var = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(var_inputs)
    #lstm_out_var = Dense(lstm_units,  kernel_regularizer=regularizers.l2(0.01), kernel_initializer='random_uniform')(var_inputs)
    attention_mul_1 = attention_3d_block_2(lstm_out_mean, 1)
    attention_mul_2 = attention_3d_block_2(lstm_out_var, 2)
    #lstm_out_var = Lambda(lambda x: 1-x)(lstm_out_var)
    attention_sum = add([attention_mul_1, attention_mul_2])
    #attention_sum = Lambda(lambda x: x/2)(attention_sum)
    #var_mul = multiply([attention_mul, lstm_out_var])
    #attention_sum = Flatten()(attention_sum)
    output_1 = Dense(40, activation='relu')(attention_sum)
    output = Dense(1, activation='sigmoid', kernel_initializer='random_uniform')(output_1)
    model = Model(input=[mean_inputs, var_inputs], output=output)
    model.summary()
    return model



def lstm_attention_variance_6(INPUT_DIM, TIME_STEPS):
    K.clear_session()
    i = Input(shape=(12,7), dtype='float32')
    enc = Bidirectional(LSTM(5, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(i)    
    dec = attention_3d_block_4(enc,1)
    # output = Dense(40, activation='relu')(dec)
    output = Dense(1, activation='sigmoid')(dec)
    model = Model(input=[i], output=output)
    model.summary()
    return model

# encoder--decoder
def lstm_attention_variance_8(INPUT_DIM, TIME_STEPS):
    K.clear_session()
    # uncertainty aware model
    i1 = Input(shape=(12, 7))
    i2 = Input(shape=(12, 1))
    
    encoder = LSTM(12)(i1)
    decoder = LSTM(12, return_sequences=True)(i2, initial_state=[encoder, encoder])
    output = Dense(1, activation="sigmoid")(decoder)
    
    #decoder = TimeDistributed(Dense(12, activation="softmax"))(decoder)
    #output = Dense(1, activation='sigmoid')(decoder)

    
    model = Model(input=[i1, i2], output=output)
    model.summary()
    return model


# encoder-attention-decoder

def lstm_attention_variance_9(INPUT_DIM, TIME_STEPS):
    K.clear_session()
    # uncertainty aware model
    i1 = Input(shape=(12, 7))
    i2 = Input(shape=(12, 1))
    
    encoder = LSTM(12, return_sequences=True)(i1)
    encoder_last = encoder[:,-1,:]
    
    decoder = Embedding(2, 12, input_length=OUTPUT_LENGTH, mask_zero=True)(i2)
    decoder = LSTM(12, return_sequences=True)(i2, initial_state=[encoder_last, encoder_last])
    
    attention = dot([decoder, encoder], axes=[2, 2])
    attention = Activation('softmax', name='attention')(attention)
    context = dot([attention, encoder], axes=[2,1])
    decoder_combined_context = concatenate([context, decoder])


    output = Dense(1, activation="sigmoid")(decoder_combined_context)
    
    #decoder = TimeDistributed(Dense(12, activation="softmax"))(decoder)
    #output = Dense(1, activation='sigmoid')(decoder)

    
    model = Model(input=[i1, i2], output=output)
    model.summary()
    return model


# attention learned from mean then applied to both mean & variance and context vectors are summed up
def lstm_attention_variance_2(INPUT_DIM, TIME_STEPS):
    # uncertainty aware model
    mean_inputs = Input(batch_shape=(None, 12, 7))
    var_inputs =  Input(batch_shape=(None, 12, 5))
    lstm_units = 12
    lstm_out_mean = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(mean_inputs)    
    lstm_out_var = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), 'ave')(var_inputs)
    #lstm_out_var = Dense(lstm_units,  kernel_regularizer=regularizers.l2(0.01), kernel_initializer='random_uniform')(var_inputs)
    attention_mul_1 = attention_3d_block_3(lstm_out_mean, lstm_out_var, 1, 2)
    #lstm_out_var = Lambda(lambda x: 1-x)(lstm_out_var)
    #attention_sum = add([attention_mul_1, attention_mul_2])
    #var_mul = multiply([attention_mul, lstm_out_var])
    #attention_sum = Flatten()(attention_mul_1)
    output_1 = Dense(40, activation='relu')(attention_mul_1)
    output = Dense(1, activation='sigmoid', kernel_initializer='random_uniform')(output_1)
    model = Model(input=[mean_inputs, var_inputs], output=output)
    return model

#%%
def output_format_2(data, single, label): # single indicates single output or time distributed output
    if single == 1:
        y = data.drop_duplicates(subset='code', keep = 'last')[label].values
        y_2 = np.reshape(y, (len(y), 1))
        return y_2
    elif single == 0:
        y = data[label].values
        y= np.reshape(y, (len(data.code.unique()), 12, 1))
        return y 

#%%
from  keras.callbacks import EarlyStopping, ModelCheckpoint
def train_generator(train_set, label, mean_features, var_features=0, batch_size=10, 
                    single_attention_vital=0, single_y=1, d='_mean', ts=12, v_len =1):
    #sequence_length=1
    while True: 
        
        # sample for each batch
        if v_len == 1:
            sequence_length = np.random.randint(1, 13)
            dataset = train_set.loc[train_set.len_sequence==sequence_length] 
        else:
            dataset = train_set
        y_train = output_format_2(dataset, single_y, label)
        
        random_idx = tuple(random.sample(range(len(y_train)), batch_size))
        y_train_feat = np.array([y_train[i] for i in random_idx])
        
        
        # reshape features 
        if (single_attention_vital==0)&(var_features==0):
            x_train_mean = reshape_features(dataset, mean_features, ts)
            x_train_feat_mean = np.array([x_train_mean[i] for i in random_idx])
            yield [x_train_feat_mean], y_train_feat
            
        elif (single_attention_vital ==1)&(var_features==0):
            feat={}
            for m in mean_features:
                feat_ = reshape_features(dataset, [m], ts)
                sub_feat = np.array([feat_[i] for i in random_idx] )
                feat[m] = sub_feat 
                
            input_y = []
            for m in mean_features:
                input_y = input_y + [feat[m]]
            yield input_y, y_train_feat
           
        elif (single_attention_vital==0) & (var_features!=0):
            x_train_var = reshape_features(dataset, var_features, ts)
            x_train_feat_var = np.array([x_train_var[i] for i in random_idx])
            x_train_mean = reshape_features(dataset, mean_features, ts)
            x_train_feat_mean = np.array([x_train_mean[i] for i in random_idx])
            yield [x_train_feat_mean,  x_train_feat_var], y_train_feat
        
        elif (single_attention_vital==1)&(var_features!=0):
            feat={}
            for m in mean_features:
                feat_ = reshape_features(dataset, [m], ts)
                sub_feat = np.array([feat_[i] for i in random_idx] )
                feat[m] = sub_feat 

            featv={}
            for v in var_features:
                featv_ = reshape_features(dataset, [v], ts)
                sub_feat = np.array([featv_[i] for i in random_idx] )
                featv[v] = sub_feat 
            t ='_var'
            yield [feat['HR'+d], feat['RR'+d], feat['SBP'+d], feat['SPO2'+d],
                   feat['TEMP'+d], feat['avpu'+d], feat['supplemental_oxygen'+d],
                   featv['HR'+t], featv['RR'+t], featv['SBP'+t], featv['SPO2'+t],
                   featv['TEMP'+t]] , y_train_feat
                   
        elif (single_attention_vital ==2):
            vitals = ['HR', 'RR', 'SBP', 'SPO2', 'TEMP']
            m='_mean'
            v='_var'
            sub_feat=[]
            for vit in vitals:
                x_feat = reshape_features(dataset, [vit+m, vit+v], ts)
                sub_feat = sub_feat + [np.array([x_feat[i] for i in random_idx])]
            
            vitals =['avpu', 'supplemental_oxygen']
            for vit in vitals:
                x_feat = reshape_features(dataset, [vit], ts)
                sub_feat = sub_feat + [np.array([x_feat[i] for i in random_idx])]

            yield sub_feat , y_train_feat
            
            
        if v_len ==1:
            if sequence_length < ts:
                sequence_length+=1
            elif sequence_length == ts:
                sequence_length=1
                
                
                
def train_generator_2(train_set, label, mean_features, var_features=0, batch_size=10, 
                    single_attention_vital=0, single_y=1, d='_mean', ts=12, v_len =1):
    while True:
        #for c in train_set.code.unique(): # for each data point in the dataset
        for c in range(0, 12):
            start = 0
            n_batch = int(np.ceil(len(train_set.loc[train_set.len_sequence==12-c])/12/128))
            for start in range(0, n_batch):
                st = start*12*128
                if start == n_batch -1 :
                    end = len(train_set.loc[train_set.len_sequence==12-c])
                else:
                    end = st + (12*128)
                                    
                dataset=train_set.loc[train_set.len_sequence==12-c].iloc[st:end]
            
                y_train = output_format_2(dataset, single_y, label)
    
                y_train_feat = y_train
                
                # reshape features 
                if (single_attention_vital==0)&(var_features==0):
                    x_train_mean = reshape_features(dataset, mean_features, ts)
                    x_train_feat_mean = np.array(x_train_mean)
                    yield [x_train_feat_mean], y_train_feat
                    
                elif (single_attention_vital ==1)&(var_features==0):
                    feat={}
                    for m in mean_features:
                        feat_ = reshape_features(dataset, [m], ts)
                        sub_feat = np.array(feat_ )
                        feat[m] = sub_feat 
                        
                    input_y = []
                    for m in mean_features:
                        input_y = input_y + [feat[m]]
                    yield input_y, y_train_feat
                   
                elif (single_attention_vital==0) & (var_features!=0):
                    x_train_var = reshape_features(dataset, var_features, ts)
                    x_train_feat_var = np.array(x_train_var)
                    x_train_mean = reshape_features(dataset, mean_features, ts)
                    x_train_feat_mean = np.array(x_train_mean)
                    yield [x_train_feat_mean,  x_train_feat_var], y_train_feat
                
                elif (single_attention_vital==1)&(var_features!=0):
                    feat={}
                    for m in mean_features:
                        feat_ = reshape_features(dataset, [m], ts)
                        sub_feat = np.array(feat_ )
                        feat[m] = sub_feat 
        
                    featv={}
                    for v in var_features:
                        featv_ = reshape_features(dataset, [v], ts)
                        sub_feat = np.array(featv_ )
                        featv[v] = sub_feat 
                    t ='_var'
                    yield [feat['HR'+d], feat['RR'+d], feat['SBP'+d], feat['SPO2'+d],
                           feat['TEMP'+d], feat['avpu'+d], feat['supplemental_oxygen'+d],
                           featv['HR'+t], featv['RR'+t], featv['SBP'+t], featv['SPO2'+t],
                           featv['TEMP'+t]] , y_train_feat  
                start=end


from numpy.random import seed
from tensorflow import set_random_seed
from keras.utils.vis_utils import plot_model as plot_model_2
from tensorflow.python.keras.callbacks import TensorBoard
from time import time

def train_models(models, figs_path, i, f, results={}, history={}, many=0, single_y = 1, ts =12, v_len=1):
    

    epochs=100   # 500 for labs
    batch_size = 128 # 128 for labs, 128 for vitalsfnews
    for key in models:
        print(key)
        mean_features = models[key]['mean_features']
        if models[key]['variance'] ==1:
            var_features=models[key]['var_features']
        else:
            var_features=0
        s = models[key]['single_attention_vital']
        v = models[key]['variance']
        # start tensorflow session for reproducibility 
        seed(1)
        set_random_seed(2)

        # Get model name
        print('\n'+ key)
        m=models[key]['model']
        
        # get initial model weights 
        #m.load_weights(figs_path+models[key]['architec']+"_"+str(ts)+"_initial_weights.h5")
        figs_path = figs_path+str(i)+'_'+f+'_'
        
        # for vital signs 
        m.compile(optimizer='adam',  loss='binary_crossentropy', metrics=['binary_accuracy'])
        
        # for labs
        #opt = SGD(lr=0.001)
        #m.compile(optimizer=opt,  loss='binary_crossentropy', metrics=['accuracy'])
        #print(m.summary())
         #plot_model_2(m, to_file=figs_path+key+'.png', show_layer_names=True)
        file_name = key+'_'+str(epochs)+'_'+str(batch_size)+'_'+str(ts)+'.h5'
        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
        checkpointer = ModelCheckpoint(filepath=figs_path+file_name, verbose=0, save_best_only=True)
        #tensorboard = TensorBoard(log_dir = figs_path+"logs/{}".format(time()))

        callbacks_def= [earlyStopping, checkpointer]
        d = models[key]['features_descrip']
        print(list(models[key]['val_set']))
        X_val_mean = reshape_features(models[key]['val_set'], mean_features, ts)
        if (s == 1)&(v==0):
            input_feat = []
            for feat in mean_features:
                input_feat = input_feat +  [np.array(reshape_features(models[key]['val_set'], [feat], ts))]

        elif(s==0)&(v==0):
            input_feat = X_val_mean
        elif (s==0)&(v==1):
            X_val_var = reshape_features(models[key]['val_set'], var_features, ts)
            input_feat = [X_val_mean, X_val_var]
        elif (s==1)&(v==1):
            input_feat = []
            for feat in mean_features:
                input_feat = input_feat +  [np.array(reshape_features(models[key]['val_set'], [feat], ts))]
            
            for feat in var_features:
                input_feat = input_feat +  [np.array(reshape_features(models[key]['val_set'], [feat], ts))]
                
                
            sub_feat = []
            for feat in mean_features:
                sub_feat = sub_feat +  [np.array(reshape_features(models[key]['train_set'], [feat], ts))]
            
            for feat in var_features:
                sub_feat = sub_feat +  [np.array(reshape_features(models[key]['train_set'], [feat], ts))]
        elif (s==2):
            v='_var'
            d='_mean'
            i1 = np.array(reshape_features(models[key]['val_set'], ['HR'+d, 'HR'+v], ts))
            i2 = np.array(reshape_features(models[key]['val_set'], ['RR'+d, 'RR'+v], ts))
            i3 = np.array(reshape_features(models[key]['val_set'], ['SBP'+d, 'SBP'+v], ts))
            i4 = np.array(reshape_features(models[key]['val_set'], ['SPO2'+d, 'SPO2'+v], ts))
            i5 =np.array(reshape_features(models[key]['val_set'], ['TEMP'+d, 'TEMP'+v], ts))
            i6 = np.array(reshape_features(models[key]['val_set'], ['avpu'+d], ts))
            i7 = np.array(reshape_features(models[key]['val_set'], ['supplemental_oxygen'+d], ts) )          
            
            input_feat = [i1,i2, i3, i4, i5, i6, i7]

            
        elif models[key]['variance']==2:
            input_feat =models[key]['input_val']
        
        single_y = models[key]['y_format']
        label = models[key]['label']
        Y_val = output_format_2(models[key]['val_set'], single_y, label)
        
        # training set
        vitals = ['HR', 'RR', 'SBP', 'SPO2', 'TEMP']
       
        #sub_feat=[]
        #for vit in vitals:
        #    x_feat = reshape_features(models[key]['train_set'], [vit+'_mean'], ts)
        #    sub_feat = x_feat
            
        #vitals =['avpu', 'supplemental_oxygen']
        #for vit in vitals:
        #    x_feat = reshape_features(models[key]['train_set'], [vit], ts)
        #    sub_feat = sub_feat + x_feat
        
        y_train_feat = output_format_2(models[key]['train_set'], single_y, label)
        #sub_feat=reshape_features(models[key]['train_set'], mean_features, ts)
        # steps_per_epoch = 50 with vitals , 200 with labs
        class_weight={0:10, 1:1}
        hist = m.fit_generator(train_generator(models[key]['train_set'], label,
                                               mean_features, var_features, batch_size, 
                                               s, single_y, d, ts=ts, v_len=v_len),
                        validation_data = (input_feat, Y_val), steps_per_epoch=48, epochs=epochs, verbose=1, 
                                               callbacks=callbacks_def)
    
        #hist = m.fit(sub_feat, y_train_feat,   
        #             validation_data = (input_feat, Y_val),epochs=epochs, verbose=1,   callbacks=callbacks_def)
        # steps per epoch = 308 for train_generator_2 
                    
        #m.save(figs_path+file_name)  
        del m
        if key in results:
            results[key].append(figs_path+key+'_'+str(epochs)+'_'+str(batch_size)+'_'+str(ts)+'.h5')
            history[key].append(hist) 
        else:
            if many==1:
                results[key]= [figs_path+key+'_'+str(epochs)+'_'+str(batch_size)+'_'+str(ts)+'.h5']
                history[key]= [hist]
            else:
                results[key]= figs_path+key+'_'+str(epochs)+'_'+str(batch_size)+'_'+str(ts)+'.h5'
                history[key]= hist
    return results, history

#%%
def time_dist_y(Y, TIME_STEPS):
    Y_train_td=[]
    for i in range(0, len(Y)):    
        Y_train_td.append(np.repeat(Y[i], TIME_STEPS))
    return np.array(Y_train_td).reshape(len(Y), TIME_STEPS, 1)

#%%



def model_selection(models, set_trained_models, set_history, roc_auc_score, ts=12, v_len=1,m_feat='HR_mean'):
    best_models = {}
    best_history= {}
    best_rocs ={}
    best_prc={}
    for key in models:
        mean_features = models[key]['mean_features']
        if models[key]['variance'] == 0:
            var_features = 0
        else:
            var_features = models[key]['var_features']
            
        single_attention_vital = models[key]['single_attention_vital']
        
        print(key)
        temp_set = set_trained_models[key]
        val_set = models[key]['val_set']        
        single_y = models[key]['y_format']
        label = models[key]['label']
        Y_val = output_format_2(models[key]['val_set'], single_y, label)
        rocs = []
        prc=[]
        for i in range(0, len(temp_set)):
            m = load_model(temp_set[i])            
            Y_pred, Y_val, len_seq, _, _,_ = test_batches(val_set, mean_features, m, var_features,
                                                        models[key]['variance'],single_attention_vital, models[key]['features_descrip'], ts=ts, v_len=v_len, m_feat = m_feat)
            Y_pred_proc = Y_pred
            #Y_pred_proc = Y_pred.flatten()
            rocs.append(roc_auc_score(Y_val, np.array(Y_pred_proc)))
            prc.append(average_precision_score(Y_val, np.array(Y_pred_proc)))
            
        idx_best = np.argmax(rocs)
        best_rocs[key] = rocs[idx_best]
        best_prc[key] = prc[idx_best]
        best_models[key]= temp_set[idx_best]
        best_history[key]= set_history[key][idx_best]

    return best_models, best_history, best_rocs, best_prc


def auroc_len_seq(compare_df, roc_auc_score):
    rocs=[]
    perc = []
    for i in range(1, 13):
        print('Length of sequence= ' + str(i)) 
        sub_test = compare_df.loc[compare_df.len_seq == i]
        num_len_seq = len(sub_test)
        perc = np.append(perc, num_len_seq/len(compare_df))
        roc = roc_auc_score(sub_test.Y_test, sub_test.Y_pred)
        rocs = np.append(rocs, roc)
        print('AUROC=' + str(roc))
    
    weighted_roc = np.dot(perc,rocs)
    print('Weighted AUROC = ' + str(weighted_roc))
    return rocs


def scale_dataset(dataset, mean_features, var_features, scaler_mean=0, scaler_var=0):
    dataset = dataset.reset_index()
    dataset_copy = dataset.copy()
    dataset_copy = dataset_copy.dropna(subset=mean_features)
    if scaler_mean == 0:
        scaler_mean = preprocessing.MinMaxScaler(feature_range=(-1,1))
        scaler_var = preprocessing.MaxAbsScaler()
        dataset_copy[mean_features]=scaler_mean.fit_transform(dataset_copy[mean_features])
        dataset_copy[var_features]=scaler_var.fit_transform(dataset_copy[var_features])
    else:
        dataset_copy[mean_features]=scaler_mean.transform(dataset_copy[mean_features])
        dataset_copy[var_features]=scaler_var.transform(dataset_copy[var_features])
    
    dataset.loc[dataset_copy.index, mean_features+var_features] = dataset_copy[mean_features+var_features].values
    dataset[mean_features+var_features]=dataset[mean_features+var_features].fillna(-5)
    return dataset, scaler_mean, scaler_var


def scale_feat(dataset, feat, func, prior=0):
    dataset = dataset.reset_index()
    dataset_copy = dataset.copy()
    dataset_copy = dataset_copy.dropna(subset=feat)
    if prior == 0:
        dataset_copy[feat]=func.fit_transform(dataset_copy[feat])
    else:
        dataset_copy[feat]=func.transform(dataset_copy[feat])
        
    dataset.loc[dataset_copy.index, feat] = dataset_copy[feat].values
    dataset[feat]=dataset[feat].fillna(-100)
    
    return dataset, func


def extract_windows(dataset):
    abnormal_ids = dataset.loc[dataset.hrs_to_firstevent<=24].drop_duplicates(subset=('hadm_id', 'point')).code.values
    all_ids = dataset.drop_duplicates(subset=('code')).code.values
    i = set(all_ids).intersection(set(abnormal_ids))
    normal_ids = list(set(all_ids).difference(i))
    abnormal_windows = dataset.loc[dataset.code.isin(abnormal_ids)]
    normal_windows = dataset.loc[dataset.code.isin(normal_ids)]
    
    print('Number of event windows=' + str(len(abnormal_windows.drop_duplicates(subset=('code')))))
    print('Number of nonevent windows=' + str(len(normal_windows.drop_duplicates(subset=('code')))))
    return normal_windows, abnormal_windows

def code_points(dataset):
    dataset['code'] = dataset['hadm_id'].map(str)+dataset['point'].map(str)
    return dataset

import random 

def prep_features_set(abnormals, normals):
    abnormals = code_points(abnormals)
    normals = code_points(normals)
    abn_nonevent_windows, abn_event_windows = extract_windows(abnormals)
    max_events = len(abn_event_windows.drop_duplicates(subset=('hadm_id', 'point')))
    norm_nonevent_windows, norm_event_windows = extract_windows(normals)
    # sub_nonevent_ids= random.sample(list(norm_nonevent_windows.code.unique()), max_events)
    sub_nonevent_ids= random.sample(list(norm_nonevent_windows.code.unique()), 18000)
    nonevent_windows = normals.loc[normals.code.isin(sub_nonevent_ids)]
    return pd.concat([abn_event_windows, nonevent_windows])

def format_labels(train_sub, mask_value, feat ='HR_mean'):
    y=train_sub.loc[train_sub[feat]!= mask_value].drop_duplicates(subset=('code'), keep='last').hrs_to_firstevent<=24
    return y

def reshape_features(dataset, features, s):
    return np.reshape(dataset[features].values, (int(len(dataset)/s), s, len(features)))
    #return np.reshape(dataset[features].values, (int(len(dataset)/s), len(features), s))


def number_points(dataset):
    t_base = pd.to_datetime('2000-01-01 00:00:00')
    dataset['charttime_sec'] = (pd.to_datetime(dataset['charttime']) - t_base).dt.total_seconds()
    dataset['point'] = dataset.groupby('hadm_id')['charttime_sec'].rank(method='first').map(int)
    return dataset

def pad_avpu_oxygen(set_features, original_dataset):
    unique_codes = set_features.code.unique()
    for i in range(0, len(unique_codes)): 
        print(i)
        code_1 = str(unique_codes[i])
        hadm_id_1 = set_features.loc[set_features.code==code_1].hadm_id.unique()[0]
        subset_1 = original_dataset.loc[original_dataset.hadm_id==hadm_id_1][['hrs_from_adm', 'hrs_to_firstevent', 'avpu', 'supplemental_oxygen']]
        subset_2 = set_features.loc[(set_features.code==code_1)&(set_features.HR_mean.notnull())][['hrs_to_firstevent', 'len_sequence']]
        point_hrs_from_adm = original_dataset.loc[original_dataset.code==code_1].hrs_from_adm.values
        len_sequence = subset_2.iloc[0].len_sequence
        subset_2['hrs_from_adm']=np.linspace(point_hrs_from_adm-(len_sequence*2)+2, point_hrs_from_adm, len_sequence)
        subset = subset_1.append(subset_2).sort_values(['hrs_from_adm'], ascending=True)
        subset[['avpu', 'supplemental_oxygen']]=subset[['avpu', 'supplemental_oxygen']].fillna(method='pad')
        set_features.loc[subset_2.index, ['avpu', 'supplemental_oxygen', 'hrs_from_adm']] = subset.loc[subset.index.isin(subset_2.index)][['avpu', 'supplemental_oxygen', 'hrs_from_adm']]
    return set_features


def missing_avpu_oxygen(dataset, keys):
    for k in keys:
        dataset.loc[(dataset.HR_mean.notnull())&(dataset[k[0]].isnull()), [k[0]]]=k[1]
    return dataset

def f(x, pop):
    
    n = int(12 - len(x)) 
    print(n)
    if n != 0:
        rows = n*[[np.nan]*len(list(x))]
        new_df = pd.DataFrame(rows, columns =list(x))
        x = pd.concat([new_df, x])
        x[['supplemental_oxygen']] = x[['supplemental_oxygen']].fillna(0)
        x[['supplemental_oxygen_linear']] = x[['supplemental_oxygen_linear']].fillna(0)
        x[['supplemental_oxygen_prev']] = x[['supplemental_oxygen_prev']].fillna(0)
        x[['supplemental_oxygen_nearest']] = x[['supplemental_oxygen_nearest']].fillna(0)
        x[['avpu']] = x[['avpu']].fillna(1)
        x[['avpu_linear']] = x[['avpu_linear']].fillna(1)
        x[['avpu_prev']] = x[['avpu_prev']].fillna(1)
        x[['avpu_nearest']] = x[['avpu_nearest']].fillna(1)
        x[['len_sequence', 'point', 'code', 'hadm_id']] = x[['len_sequence', 'point', 'code', 'hadm_id']].fillna(method='bfill')
        x[['HR_mean', 'RR_mean', 'SBP_mean', 'TEMP_mean', 'SPO2_mean']] = x[['HR_mean', 'RR_mean', 'SBP_mean', 'TEMP_mean', 'SPO2_mean']].fillna(pop[['HR_mean', 'RR_mean', 'SBP_mean', 'TEMP_mean', 'SPO2_mean']].mean())
        x[['HR_linear', 'RR_linear', 'SBP_linear', 'TEMP_linear', 'SPO2_linear']] = x[['HR_linear', 'RR_linear', 'SBP_linear', 'TEMP_linear', 'SPO2_linear']].fillna(pop[['HR_linear', 'RR_linear', 'SBP_linear', 'TEMP_linear', 'SPO2_linear']].mean())
        x[['HR_prev', 'RR_prev', 'SBP_prev', 'TEMP_prev', 'SPO2_prev']] = x[['HR_prev', 'RR_prev', 'SBP_prev', 'TEMP_prev', 'SPO2_prev']].fillna(pop[['HR_prev', 'RR_prev', 'SBP_prev', 'TEMP_prev', 'SPO2_prev']].mean())
        x[['HR_var', 'RR_var', 'SBP_var', 'TEMP_var', 'SPO2_var']] = x[['HR_var', 'RR_var', 'SBP_var', 'TEMP_var', 'SPO2_var']].fillna(pop[['HR_var', 'RR_var', 'SBP_var', 'TEMP_var', 'SPO2_var']].max())
        x['index'] = np.arange(0, 12)
        return (x)
    else:
        x['index'] = np.arange(0, 12)
        return (x)


def pre_pad(dataset_pr, train_set):
    temp_set = dataset_pr.loc[dataset_pr.HR_mean.notnull()]
    #temp_set = temp_set.drop(['index'], axis=1)
    temp_set = temp_set.groupby('code').apply(f, train_set).reset_index(drop=True)
    return temp_set


def pre_pad2(dataset_pr):
    dataset_pr[[ 'supplemental_oxygen_linear', 'supplemental_oxygen_prev', 'supplemental_oxygen_nearest']]= dataset_pr[[ 'supplemental_oxygen_linear', 'supplemental_oxygen_prev', 'supplemental_oxygen_nearest']].fillna(0)
    print(1)
    dataset_pr[['avpu_linear', 'avpu_nearest', 'avpu_prev']]= dataset_pr[['avpu_linear', 'avpu_nearest', 'avpu_prev']].fillna(1)
    print(2)
    dataset_pr[['HR_mean', 'RR_mean', 'SBP_mean', 'TEMP_mean', 'SPO2_mean']] = dataset_pr[['HR_mean', 'RR_mean', 'SBP_mean', 'TEMP_mean', 'SPO2_mean']].fillna(dataset_pr[['HR_mean', 'RR_mean', 'SBP_mean', 'TEMP_mean', 'SPO2_mean']].mean())
    print(3)
    dataset_pr[['HR_linear', 'RR_linear', 'SBP_linear', 'TEMP_linear', 'SPO2_linear']] = dataset_pr[['HR_linear', 'RR_linear', 'SBP_linear', 'TEMP_linear', 'SPO2_linear']].fillna(dataset_pr[['HR_linear', 'RR_linear', 'SBP_linear', 'TEMP_linear', 'SPO2_linear']].mean())
    print(4)
    dataset_pr[['HR_prev', 'RR_prev', 'SBP_prev', 'TEMP_prev', 'SPO2_prev']] = dataset_pr[['HR_prev', 'RR_prev', 'SBP_prev', 'TEMP_prev', 'SPO2_prev']].fillna(dataset_pr[['HR_prev', 'RR_prev', 'SBP_prev', 'TEMP_prev', 'SPO2_prev']].mean())
    print(5)
    dataset_pr[['HR_var', 'RR_var', 'SBP_var', 'TEMP_var', 'SPO2_var']] = dataset_pr[['HR_var', 'RR_var', 'SBP_var', 'TEMP_var', 'SPO2_var']].fillna(dataset_pr[['HR_var', 'RR_var', 'SBP_var', 'TEMP_var', 'SPO2_var']].max())
    return dataset_pr

def pre_pad3(dataset_pr, feature_sets):
    
    for k in feature_sets:
        print(k)
        for m in feature_sets[k]['mean_features']:
            print(m)
            if 'avpu' in m:
                dataset_pr[m] = dataset_pr[m].fillna(1)
            elif 'supplemental_oxygen' in m:
                dataset_pr[m] = dataset_pr[m].fillna(0)
            else:
                dataset_pr[m] = dataset_pr.groupby("code").transform(lambda x: x.fillna(x.mean()))
    
    var_features = feature_sets['GP']['var_features']
    for m in var_features:
        dataset_pr[m] = dataset_pr.groupby("code").transform(lambda x: x.fillna(x.max()))
    return dataset_pr


def pre_pad4(df):
    df[[ 'supplemental_oxygen_linear', 'supplemental_oxygen_prev', 'supplemental_oxygen_nearest', 'supplemental_oxygen']]= df[[ 'supplemental_oxygen_linear', 'supplemental_oxygen_prev', 'supplemental_oxygen_nearest', 'supplemental_oxygen']].fillna(0)
    df[['avpu_linear', 'avpu_nearest', 'avpu_prev', 'avpu']]= df[['avpu_linear', 'avpu_nearest', 'avpu_prev', 'avpu']].fillna(1)
    
    feat = ['HR_mean', 'RR_mean', 'SBP_mean', 'TEMP_mean', 'SPO2_mean','HR_linear', 'RR_linear', 'SBP_linear', 'TEMP_linear', 'SPO2_linear','HR_prev', 'RR_prev', 'SBP_prev', 'TEMP_prev', 'SPO2_prev',
            'HR_var', 'RR_var', 'SBP_var', 'TEMP_var', 'SPO2_var']
    df[feat] = df[feat].fillna(df.groupby('code').transform('mean'))
        
    return df 

from scipy import stats

#%% Evaluate performance using bootstrapping 
def performance_bootstrap(nb, test_set, key, metrics, r):

    r= int((r/100)*len(test_set))
    scores_key=[]
    if len(test_set) > r:
        for i in range(0, nb):
            temp_set = test_set.loc[np.random.choice(test_set.index, r, replace = False)]
            Y_test = temp_set.label.values
            if len(np.unique(Y_test))<2:
                continue
            
            Y_pred_proc = temp_set[key].values
            if key =='NEWS': 
                thresh = 0.25
            else: 
                thresh = 0.4
            
            Y_pred_bin = Y_pred_proc>=thresh
            
            scores =[]
            for m in metrics:
                metric = m
                func = metrics[m]
                if (metric == 'ACC') | (metric== 'F1 score') | (metric=='SENS') | (metric=='SPEC') | (metric=='PPV'):
                    scores.append(round(func(Y_test, Y_pred_bin),3))
                else:
                    scores.append(round(func(Y_test, Y_pred_proc),3))
            scores_key.append(scores)
        performance_metrics=pd.DataFrame(data=scores_key, columns=list(metrics.keys()))
        new_cols=[]
        row=[]
        for n in list(performance_metrics):
            trials = performance_metrics[n].values
            #plt.hist(trials, bins=30,  density=True, stacked=True)
            #plt.title(n)
            #plt.show()
            new_cols.append(n+'_mean')
            temp_mean = trials.mean()
            temp_std = trials.std()
            row.append(temp_mean)
            new_cols.append(n+'_25CI')
            alpha = 1 - 0.95
            # (2) Get the z values
            UBP = 1- (alpha/2)
            LBP = alpha/2 
            z_UBP = stats.norm.ppf(UBP)
            z_LBP = stats.norm.ppf(LBP)       # note this will return a negative value 
            # (3) calculate the lower and upper bounds
            UB = temp_mean + z_UBP*(temp_std/np.sqrt(nb))
            LB = temp_mean + z_LBP*(temp_std/np.sqrt(nb))  
            row.append(LB)
            row.append(UB)
            new_cols.append(n+'_75CI')
    
        new_metrics = pd.DataFrame(data = [[key]+row], columns = ['model']+new_cols)
        return new_metrics
    else:
        return pd.DataFrame( columns=list(metrics.keys()))

def prob_hour_to_event(results_testset, figs_path, temp_keys_2, colors):
    #mpl.rcParams['font.size'] = 23
    abnormals_windows_df = results_testset.loc[results_testset.label==1]
    abnormals_windows_df['hrs_to_firstevent_2']=np.round(abnormals_windows_df.hrs_to_firstevent,0)
    abnormals_results_mean = abnormals_windows_df.groupby('hrs_to_firstevent_2').mean()
    #abnormals_results_std = abnormals_windows_df.groupby('hrs_to_firstevent_2').std()
    for n in temp_keys_2:    
        #plt.errorbar(abnormals_results_mean.index, abnormals_results_mean[n], abnormals_results_std[n], label=n)
        if n== 'GP_UA-LSTM-ATT-2':
            l='DEWS'
        elif n=='NEWS':
            l = 'Normalized NEWS'
        plt.plot(abnormals_results_mean.index, abnormals_results_mean[n], colors[n], label=l, linewidth=2.6)
    
    plt.xlim(24,-1)
    plt.ylim(0,1)
    plt.xlabel('Hours to Outcome')
    #plt.title('(b)')
    #plt.ylabel('Mean probability of event')
    plt.legend(fontsize=18, loc='lower left')
    plt.plot([0, 0], [0,1], 'k--', linewidth=2.6)
    plt.grid(True)
    #plt.savefig(figs_path+'prob_hour_event.png', dpi=200, bbox_inches='tight')
    #plt.show()
    
def prob_hour_from_adm(results_testset, figs_path, temp_keys_2, colors):
    #mpl.rcParams['font.size'] = 20
    normals_windows_df = results_testset.loc[results_testset.hrs_to_firstevent.isnull()]
    normals_windows_df['hrs_from_adm_2'] = np.round(normals_windows_df.hrs_from_adm, 0)
    
    
    normals_results_mean = normals_windows_df.groupby('hrs_from_adm_2').mean()
    #normals_results_mean = normals_results_mean.loc[normals_results_mean.index%2 == 0]
    for n in temp_keys_2:    
        #plt.errorbar(abnormals_results_mean.index, abnormals_results_mean[n], abnormals_results_std[n], label=n)
        if n== 'GP_UA-LSTM-ATT-2':
            l='DEWS'
        elif n=='NEWS':
            l = 'Normalized NEWS'
        plt.plot(normals_results_mean.index, normals_results_mean[n],colors[n], label=l, linewidth=2.6)
    
    plt.ylim(0,1)
    plt.xlim(0, 120)
    plt.xlabel('Hours from Admission')
    #plt.title('(a)')
    #plt.ylabel('Mean probability of event')
    plt.legend(loc='upper right', fontsize=18)
    plt.plot([24, 24], [0,1], 'k--', linewidth=2.6)
    plt.grid(True)
    #plt.savefig(figs_path+'prob_from_adm_0_150.png', dpi=200, bbox_inches='tight')
    #plt.show()
    
def auroc_hour_to_event(results_testset, figs_path, temp_keys_2, func, colors):
    #windows_df = results_testset.loc[]
    
    plt.figure(figsize=(7,5))
    hrs = [0,2,4,6,8,10,12,14,16,18,20,22,24]

    d=0
    plt.grid(True)
    for n in temp_keys_2:
        print(n)
        auroc=[]
        for i in range(0, len(hrs)):
            hr = hrs[i]
            df = results_testset.loc[(results_testset.hrs_to_firstevent.isnull())|(results_testset.hrs_to_firstevent>=hr)]
            Y_pred = df[n]
            Y_test = df.label
            auroc.append(round(func(Y_test, Y_pred),3))
        print(np.std(auroc))
        #plt.errorbar(abnormals_results_mean.index, abnormals_results_mean[n], abnormals_results_std[n], label=n)
        if n == 'GP_UA-LSTM-ATT-2':
            n = 'DEWS'
        plt.bar([x+d for x in hrs], auroc, width=0.5,label=n, color=colors[n])
        d+=0.5
    
    plt.ylim(0.82,0.90)
    plt.xlim(26,-1)
    plt.xlabel('Hours to Outcome')
    plt.ylabel('AUROC')
    plt.legend(loc='upper left', fontsize=15)
    
    plt.savefig(figs_path+'auroc_from_adm_ports.png', bbox_inches='tight')
    plt.show()
    
figs_path = 'C:/Users/ball4624/Desktop/PhD-research-code/Year 2 (GP-DL)/Figures_Dec_2_2018/'
def attention_sample_mean(best_models, test_set_temp, X_test_id, model_name, mean_features, var_features):
    K.clear_session()
    m = load_model(best_models[model_name]) 
    attention_vectors = {}
    for i in range(0, len(X_test_id)):
        sample_code = X_test_id[i]
        temp_set = test_set_temp.loc[test_set_temp.code==sample_code]
        X_test_mean = gpdl.reshape_features(temp_set, mean_features, 12)
        X_test_var = gpdl.reshape_features(temp_set, var_features, 12)
    
        if model_name=='LSTM-ATT':
            test_input =[np.array(X_test_mean)[0]]
        else:
            test_input = [np.array(X_test_mean)[0], np.array(X_test_var)[0]]
        attention_vector = np.sum(get_activations(m, test_input,model_name, print_shape_only=True, layer_name='attention_mul_2')[0], axis=2).squeeze()
        
        pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar', ylim=(-0.20, 0.2),
                                                                         title='Sum Attention*hidden state per time step for sample input for '+str(X_test_id[i]))

        assert (np.sum(attention_vector) - 1.0) < 1e-5
        attention_vectors[X_test_id[i]]=attention_vector
    plt.savefig(figs_path+'alpha_'+model_name+'.jpg')
    return pd.DataFrame.from_dict(attention_vectors, orient='index').reset_index()

def attention_sample(best_models, test_set_temp, X_test_id, model_name, mean_features, var_features):
    K.clear_session()
    m = load_model(best_models[model_name]) 
    attention_vectors = {}
    for i in range(0, len(X_test_id)):
        sample_code = X_test_id[i]
        temp_set = test_set_temp.loc[test_set_temp.code==sample_code]
        X_test_mean = gpdl.reshape_features(temp_set, mean_features, 12)
        X_test_var = gpdl.reshape_features(temp_set, var_features, 12)
    
        if model_name=='LSTM-ATT':
            test_input =[np.array(X_test_mean)[0]]
        else:
            test_input = [np.array(X_test_mean)[0], np.array(X_test_var)[0]]
        attention_vector = get_activations(m, test_input,model_name, print_shape_only=True, layer_name='attention_mul_1')[0]
        print(attention_vector)
        pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar',
                                                                         title='Attention (%) per time step for sample input for '+str(X_test_id[i]))

        assert (np.sum(attention_vector) - 1.0) < 1e-5
        attention_vectors[X_test_id[i]]=attention_vector
       
    return pd.DataFrame.from_dict(attention_vectors, orient='index').reset_index()


def plot_attention_vitals(sample_code, attention_vectors, model_name, test_set_features_pr_2, mean_features, var_features, figs_path):
    print(sample_code)
    sample_patient = test_set_features_pr_2.loc[test_set_features_pr_2.code==sample_code]
    attend_perc= attention_vectors.loc[attention_vectors['index'] == sample_code].values[0][1:]
    x= sample_patient.hrs_from_adm.values
    x = np.array(x, dtype='float')
    
    dotcolors=[(0.0, 0.32, 0.56, a) for a in attend_perc]
    colors=['r', 'g', 'b', 'y', 'k', 'm', 'c']
    for m in range(0, len(mean_features)):
        feature = mean_features[m]
        y=sample_patient[feature].values
        if 'mean' in feature:
            var_feature = var_features[m]
            yerr=sample_patient[var_feature].values
            plt.errorbar(x, y, yerr=yerr, color=colors[m], capsize=3, marker='o', linestyle='solid', label=feature)
    
        else:
            plt.plot(x, y, colors[m]+'o-', label=feature)
    
    i = min(x)
    n=0
    while n < 12:
        plt.fill_between([x[n]-1, x[n]+1], -1.5, 1.5, facecolor=dotcolors[n])
        n+=1
        i+=2
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.xlabel('Hours from admission')
    plt.ylabel('Scaled feature value')
    plt.axis((min(x), max(x)+1, -1.1, 1.1))
    plt.savefig(figs_path+'attention_features_'+sample_code+'_'+ model_name+'.png', dpi=300, bbox_inches = 'tight')
    #plt.fill_between(x, min_y, max_y, facecolor='yellow', interpolate=True)
    plt.show()
    
    
def plot_vitals(sample_code, test_set_features_pr_2, mean_features, var_features, figs_path):
    print(sample_code)
    sample_patient = test_set_features_pr_2.loc[test_set_features_pr_2.code==sample_code]
    x= sample_patient.hrs_from_adm.values
    x= np.arange(0, 12, 1)
    x = np.array(x, dtype='float')
    
    colors=['r', 'g', 'b', 'y', 'k', 'm', 'c']
    for m in range(0, len(mean_features)):
        feature = mean_features[m]
        y=sample_patient[feature].values
        if ('avpu'  in feature) | ('supplemental_oxygen' in feature):
            plt.plot(x, y, colors[m]+'o-', label=feature)
        
        elif 'mean' in feature:
            print(m)
            var_feature = var_features[m]
            yerr=sample_patient[var_feature].values
            plt.errorbar(x, y, yerr=yerr, color=colors[m], capsize=3, marker='o', linestyle='solid', label=feature)
            
    
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.xlabel('Sampled points within window')
    plt.ylabel('Scaled feature value')
    plt.axis((min(x), max(x)+1, -1.1, 1.1))
    plt.savefig(figs_path+sample_code+'_.png', dpi=300, bbox_inches = 'tight')
    #plt.fill_between(x, min_y, max_y, facecolor='yellow', interpolate=True)
    plt.show()

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

#%%
    
def perf_bootstrap(nb, temp_keys_2, results_testset, metrics, r):
    performance_bootstrapped=pd.DataFrame()
    for i in range(0, len(temp_keys_2)):
       if i == 0:
           performance_bootstrapped = performance_bootstrap(nb,results_testset, temp_keys_2[i], metrics,r)
       else:
           performance_bootstrapped = performance_bootstrapped.append(performance_bootstrap(nb,results_testset, temp_keys_2[i], metrics, r))
    return performance_bootstrapped

def my_sensitivity(TP,FP,TN,FN):    
    return round(TP/(TP+FN), 3)

def my_specificity(TP,FP,TN,FN):    
    return round(TN/(TN+FP), 3)

def my_precision(TP,FP,TN,FN):    
    return round(TP/(TP+FP), 3)

def my_accuracy(TP,FP,TN,FN):    
    return round((TP+TN)/(TP+FN+TN+FP), 3)

def my_f1(TP,FP,TN,FN):
    return round(2*TP/((2*TP)+FP+FN), 3)


def perf_bootstrap_2(test_set, r, nb, keys, CL, metrics):
    # test_set is a dataframe of the test set
    # r is the percentage of rows/ observations to randomly select from test set per iteration
    # We set r as number of rows rather than percentage (July 12, 2018)
    # nb is the number of times to bootstrap 
    # N is an array of hours to evaluate at 
    
    # convert r to number of rows
    r = int((r/100)*len(test_set))
    
    perf={}
    thresholds={'NEWS': 0.25 , 'LR': 0.881,
                'CF_LR':0.64 , 'CF_LSTM':0.715 , 'CF_BILSTM':0.725, 'CF_LSTM-ATT-1':0.715, 'CF_LSTM-ATT-2':0.685,
                'LI_LR':0.662 , 'LI_LSTM':0.732 , 'LI_BILSTM':0.805, 'LI_LSTM-ATT-1':0.685, 'LI_LSTM-ATT-2':0.71,
                'GP_LR':0.585 , 'GP_LSTM': 0.525, 'GP_BILSTM':0.675, 'GP_LSTM-ATT-1':0.496, 'GP_LSTM-ATT-2':0.65,
                'GP_UA-LSTM-ATT-1': 0.555,'GP_UA-LSTM-ATT-2': 0.725,
                'CF-MLP': 0.661}
    # Iterate nb times 
    for k in keys:
        perf[k]={}
    for i in range (0, nb):
        print(i)
        # Take r random rows from the test set 
        np.random.seed(i)
        temp_set = test_set.loc[np.random.choice(test_set.index, r, replace = False)]
        Y_test = temp_set['label_1'].values.astype(int)
        
        for k in keys: # keys of models
            
            thresh=thresholds[k]
            Y_pred_proc = temp_set[k].values
            Y_pred_bin = Y_pred_proc>=thresh 
            
            res = pd.DataFrame()
            res['true']=Y_test
            res['pred']=Y_pred_bin
            TP = len(res.loc[(res.true==1)&(res.pred==1)])
            FP = len(res.loc[(res.true==0)&(res.pred==1)])
            TN = len(res.loc[(res.true==0)&(res.pred==0)])
            FN = len(res.loc[(res.true==1)&(res.pred==0)])
        
            if len(np.unique(Y_test))<2:
                continue
             
            for metric in metrics: # Evaluate performance
                func = metrics[metric]
                if (metric == 'ACC') | (metric== 'F1 score') | (metric=='SENS') | (metric=='SPEC') | (metric=='PPV'):
                    if metric in perf[k].keys():
                        perf[k][metric].append(func(TP,FP,TN,FN))
                    else:
                        perf[k][metric]= [func(TP,FP,TN,FN)]
                else:
                    if metric in perf[k].keys():
                        perf[k][metric].append(round(func(Y_test, Y_pred_proc),3))
                    else:
                        perf[k][metric]= [round(func(Y_test, Y_pred_proc),3)]
                        
                    
    return perf 


def perf_bootstrap_3(perf, keys, M, CL, nb):
    alpha = 1 - CL
    # (2) Get the z values
    UBP = 1- (alpha/2)
    LBP = alpha/2 
    z_UBP = stats.norm.ppf(UBP)
    z_LBP = stats.norm.ppf(LBP)       # note this will return a negative value 
    results={}
    for m in M:
        results[m+'_mean']={}
        results[m+'_interval']={}
        for k in keys:
            mean=np.mean(perf[k][m])
            std=np.std(perf[k][m])
            results[m+'_mean'][k] = round(mean,3)
            results[m+'_interval'][k] = (round(mean + z_LBP*(std/np.sqrt(nb)),3) ,round( mean + z_UBP*( std/np.sqrt(nb)), 3))
    
               #temp_aurocs[m]['std'] = np.std(temp_aurocs[m]['aurocs'])
               #temp_aurocs[m]['interval'] = (temp_aurocs[m]['mean'] + z_LBP*( temp_aurocs[m]['std']/np.sqrt(nb))  , temp_aurocs[m]['mean'] + z_UBP*( temp_aurocs[m]['std']/np.sqrt(nb)))
    return results
                
'''    # Evaluate the bootstrapping confidence interval 
    # (1) Alpha = 1 - confidence level 
    alpha = 1 - CL
    # (2) Get the z values
    UBP = 1- (alpha/2)
    LBP = alpha/2 
    z_UBP = stats.norm.ppf(UBP)
    z_LBP = stats.norm.ppf(LBP)       # note this will return a negative value 
        
    for m in M:
        temp_aurocs[m]['mean'] = np.mean(temp_aurocs[m]['aurocs'])
        temp_aurocs[m]['std'] = np.std(temp_aurocs[m]['aurocs'])
        temp_aurocs[m]['interval'] = (temp_aurocs[m]['mean'] + z_LBP*( temp_aurocs[m]['std']/np.sqrt(nb))  , temp_aurocs[m]['mean'] + z_UBP*( temp_aurocs[m]['std']/np.sqrt(nb)))
    
    return temp_aurocs'''


#%%
def add_diagnosis_groups(results_testset):
    results_testset['ICD_group']=np.zeros(len(results_testset))
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='A00')&(results_testset.diagnosis_code<='B99XXX')), 1, results_testset.ICD_group)
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='C00')&(results_testset.diagnosis_code<='D49XXX')), 2, results_testset.ICD_group)
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='D50')&(results_testset.diagnosis_code<='D89XXX')), 3, results_testset.ICD_group)
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='E00')&(results_testset.diagnosis_code<='E89XXX')), 4, results_testset.ICD_group)
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='F01')&(results_testset.diagnosis_code<='F99XXX')), 5, results_testset.ICD_group)
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='G00')&(results_testset.diagnosis_code<='G99XXX')), 6, results_testset.ICD_group)
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='H00')&(results_testset.diagnosis_code<='H59XXX')), 7, results_testset.ICD_group)
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='H60')&(results_testset.diagnosis_code<='H95XXX')), 8, results_testset.ICD_group)
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='I00')&(results_testset.diagnosis_code<='I99XXX')), 9, results_testset.ICD_group)
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='J00')&(results_testset.diagnosis_code<='J99XXX')), 10, results_testset.ICD_group)
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='K00')&(results_testset.diagnosis_code<='K95XXX')), 11, results_testset.ICD_group)
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='L00')&(results_testset.diagnosis_code<='L99XXX')), 12, results_testset.ICD_group)
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='M00')&(results_testset.diagnosis_code<='M99XXX')), 13, results_testset.ICD_group)
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='N00')&(results_testset.diagnosis_code<='N99XXX')), 14, results_testset.ICD_group)
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='O00')&(results_testset.diagnosis_code<='O9AXXX')), 15, results_testset.ICD_group)
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='P00')&(results_testset.diagnosis_code<='P96XXX')), 16, results_testset.ICD_group)
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='Q00')&(results_testset.diagnosis_code<='Q99XXX')), 17, results_testset.ICD_group)
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='R00')&(results_testset.diagnosis_code<='R99XXX')), 18, results_testset.ICD_group)
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='S00')&(results_testset.diagnosis_code<='T88XXX')), 19, results_testset.ICD_group)
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='V00')&(results_testset.diagnosis_code<='Y99XXX')), 20, results_testset.ICD_group)
    results_testset['ICD_group'] = np.where(((results_testset.diagnosis_code>='Z00')&(results_testset.diagnosis_code<='Z99XXX')), 21, results_testset.ICD_group)
    return results_testset



#%% Plot ROC curves
def plot_roc(keys, yt, yp, colors):
    for k in keys:
        y_true = yt[k]
        y_probas =  yp[k]
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_probas)
        roc_auc = metrics.auc(fpr, tpr)        
        if k == 'GP_UA-LSTM-ATT-2':
            k = 'DEWS'
        plt.plot(fpr, tpr, colors[k], label = k, linewidth=2)
    plt.legend(loc = 'lower right')
    plt.title('Receiver Operating Characteristic')
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

#%% Plot AUPRC 
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score, precision_recall_curve

    
def plot_auprc(keys, yt, yp, colors):
    for k in keys:
        y_true = yt[k]
        y_probas =  yp[k]
        precision, recall, _ = precision_recall_curve(y_true, y_probas)
        if k == 'GP_UA-LSTM-ATT-2':
            k = 'DEWS'
        plt.step(recall, precision, colors[k], where='post', label=k)
    plt.legend(loc = 'lower right')
    plt.title('Precision-Recall Curve')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()
    
    #%% Clinical utility
def sensitivity_at_thresh(df, N, thresh=np.arange(0, 1.1, 0.1)):

    sensitivity = {}
    P_dict = {}
    thresh_results = {}
    
    for key in df:
        sens = []
        p =[]
        rows = []
        for t in range(0, len(thresh)):
            df[key]['alert'] = df[key][key] >= thresh[t]            
            TP = len(df[key].loc[(df[key].alert==1)&(df[key].label==1)])
            FN = len(df[key].loc[(df[key].alert==0)&(df[key].label== 1)])
            FP = len(df[key].loc[(df[key].alert==1)&(df[key].label == 0) ])
            TN =  len(df[key].loc[(df[key].alert==0)&(df[key].label == 0 ) ])
            if (TP+FP == 0):
                PPV = -1
            else:
                PPV = np.round((TP/(TP+FP)), 2)
            
            rows = rows + [[thresh[t], TP, FN, FP, TN, PPV]]
            sens.append(np.round(TP/(TP+FN), 3))
            
            P=TP+FP
            p.append(P/len(df[key]))            

        sensitivity[key]=sens
        P_dict[key]=p
        thresh_results[key] = pd.DataFrame(data=rows, columns = ['thresh', 'TP', 'FN', 'FP', 'TN', 'PPV'])
        
    return sensitivity , P_dict, thresh_results


import matplotlib.pyplot as mpl
mpl.rcParams['font.size'] = 20

def plot_efficiency(sensitivity, positive_rate, title, colors, ct, ct1, ct2, figs_path):
    axes = plt.gca()
    M = ['GP_UA-LSTM-ATT-2', 'NEWS']
    for m in range(0, len(M)):
        key=M[m]       
        if key=='GP_UA-LSTM-ATT-2':
            
            l = 'DEWS'
        else:
            l= key
        print(l)
        plt.plot(sensitivity[key], positive_rate[key],colors[l],label=l, linewidth=2.6, markersize=4)
    
    plt.plot([0.8, 0.8], [-0.05, positive_rate['NEWS'][ct]-ct1], 'k--', linewidth=2.6)
    plt.plot([0, 0.8], [positive_rate['NEWS'][ct]-ct1, positive_rate['NEWS'][ct]-ct1], 'k--', linewidth=2.6)
    plt.plot([0, 0.8], [positive_rate['NEWS'][ct]-ct2, positive_rate['NEWS'][ct]-ct2], 'k--', linewidth=2.6)

    
    #plt.title(title)
    axes.set_xlim([0.0, 1])
    plt.grid(True)
    axes.set_ylim([0, 1])
    plt.xlabel('Sensitivity')
    plt.ylabel('Positive Alerts')
    plt.legend(fontsize=15)
    #plt.savefig(figs_path+'efficiency_curve_'+title+'.png', dpi=400, bbox_inches='tight')
    #plt.show()
    
#%% Get attention weights
    
def attention_weights_hist(model_v, models, key, figs_path, idx_code, dataset):
    inp = model_v.input                                           # input placeholder
    outputs = [layer.output for layer in model_v.layers]          # all layer outputs
    #model_v.summary()
    
    temp_set = models[key]['test_set']
    d = models[key]['features_descrip']
    y = output_format_2(models[key]['test_set'], models[key]['y_format'], models[key]['label'])
    X_test_mean =  np.reshape(temp_set[models[key]['mean_features']].values, (len(temp_set.drop_duplicates(subset='code', keep='last')),12,7))
    subset=dataset.loc[dataset.code==idx_code]
    
    if 'UA-LSTM-ATT-2' in key:
        # (1) Format input for model 
        vitals = ['HR', 'RR', 'SBP', 'SPO2', 'TEMP', 'avpu', 'supplemental_oxygen']
        input_feat = []
        for feat in vitals:
            input_feat = input_feat +  [np.reshape(reshape_features(subset, [feat+d], 12)[0], (1,12,1))]    
        input_feat_2 = []
        for feat in  ['HR', 'RR', 'SBP', 'SPO2', 'TEMP']:
            input_feat_2 = input_feat_2 + [np.reshape(reshape_features(subset, [feat+'_var'], 12)[0], (1,12,1))]
        
        for j in range(5,7):
            for i in range(0,12):
                input_feat[j][0][i] = [-1]
        
        test = np.concatenate((input_feat, input_feat_2))
        feat =['HR_mean', 'RR_mean', 'SBP_mean', 'SPO2_mean', 'TEMP_mean', 'avpu', 'supplemental_oxygen']#,
        #           'HR_var', 'RR_var', 'SBP_var', 'SPO2_var', 'TEMP_var']
    
        # (2) Get attention vectors 
        y = []
        # attention vectors 
        for f in range(0, len(feat)):
            layers=['attention_vec_'+str(f+1)]
            outputs = [model_v.get_layer(layer).output for layer in layers]          # all layer outputs
    
            functors = [K.function([*inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions
            layer_outs = [func([*test, 0]) for func in functors]
            
            if '_mean' in feat[f]:
                c = 'k'
            elif '_var' in feat[f]:
                c='silver'
            plt.figure(num=None, figsize=(4.5, 2), dpi=80)
            plt.bar(np.arange(0, 12, 1), np.array(layer_outs[0][0]).flatten(), color=c, align='center')
            y=y+[np.array(layer_outs[0][0]).flatten().tolist()]
            plt.axis((0, 11, 0, 0.8))
            plt.yticks(np.arange(0, 0.8, 0.2), fontsize=17)
            plt.xticks(np.arange(0, 12, 1), np.arange(1, 13, 1), fontsize=17)
            plt.ylabel('Attention (%)', fontsize=20, labelpad=10)
            plt.xlabel('Sampled points per window (t)', fontsize= 20)
            #plt.figure(figsize=(5,1))
            plt.savefig(figs_path+key+'code='+str(idx_code)+'_att_vis_layer='+feat[f]+'.jpg', dpi=300, bbox_inches='tight')
            plt.show()
            
        # (3) Get joined context vectors 
        layers=['add_3']
        outputs = [model_v.get_layer(layer).output for layer in layers] 
        functors = [K.function([*inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions
        layer_outs = [func([*test, 0]) for func in functors]
        plt.figure(num=None, figsize=(4.5, 2), dpi=80)
        plt.bar(np.arange(0, 5, 1), np.array(layer_outs[0][0]).flatten(), color=c, align='center')
        plt.axis((0, 5, -1 , 1))
        plt.show()
        
        # model output
        layers = ['dense_28']
        outputs = [model_v.get_layer(layer).output for layer in layers] 
        functors = [K.function([*inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions
        output = [func([*test, 0]) for func in functors]
        
        #for i in range(-17, -10):
        #    layer_num = i
        #    ax = sns.heatmap(layer_outs[layer_num][0][0], linewidths=0.05)
        #    plt.savefig(figs_path+key+'idx='+str(idx)+'_att_vis_layer='+str(layer_num)+'vital='+feat[n]+'.jpg', dpi=300)
         #   plt.show()
         #   n= n+1    
    
    

    elif 'UA-LSTM-ATT-1' in key:
        test =[np.reshape( X_test_mean[idx], (1,12,7)), np.reshape( X_test_var[idx], (1,12,5))]
        functors = [K.function([*inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions
        layer_outs = [func([*test, 0]) for func in functors]
        
        layer_num = -9
        ax = sns.heatmap(layer_outs[layer_num][0][0], linewidths=0.05)
        plt.savefig(figs_path+key+'idx='+str(idx)+'_att_vis_layer='+str(layer_num)+'mean.jpg', dpi=300)
        plt.show()
        
        layer_num = -8
        ax = sns.heatmap(layer_outs[layer_num][0][0], linewidths=0.05)
        plt.savefig(figs_path+key+'idx='+str(idx)+'_att_vis_layer='+str(layer_num)+'variance.jpg', dpi=300)
        plt.show()
    
    elif 'LSTM-ATT-1' in key:
        test = np.reshape(X_test_mean[idx], (1,12,7))
        functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions
        layer_outs = [func([test,0]) for func in functors]
        layer_num = 7
        print(layer_outs[layer_num][0])
        plt.bar(np.arange(0, 12, 1), np.array(layer_outs[layer_num][0]).flatten())
        plt.axis((0, 12, 0, 0.8))
        #ax = sns.heatmap(layer_outs[layer_num][0][0], linewidths=0.05)
        plt.savefig(figs_path+key+'idx='+str(idx)+'_att_vis_layer='+str(layer_num)+'.jpg', dpi=300)
        plt.show()
        print('y_true=' + str(y[idx]))
        print('y_pred='+ str(layer_outs[-1][0][0]))
    elif 'LSTM-ATT-2' in key:
        i1 = gpdl.reshape_features(test_set, ['HR'+d], 12)
        i2 = gpdl.reshape_features(test_set, ['RR'+d], 12)
        i3 = gpdl.reshape_features(test_set, ['SBP'+d], 12)
        i4 = gpdl.reshape_features(test_set, ['SPO2'+d], 12)
        i5 = gpdl.reshape_features(test_set, ['TEMP'+d], 12)
        i6 = gpdl.reshape_features(test_set, ['avpu'+d], 12)
        i7 = gpdl.reshape_features(test_set, ['supplemental_oxygen'+d], 12)
        
        i1 = np.reshape(i1[idx], (1,12,1))
        i2 = np.reshape(i2[idx], (1,12,1))
        i3 = np.reshape(i3[idx], (1,12,1))
        i4 = np.reshape(i4[idx], (1,12,1))
        i5 = np.reshape(i5[idx], (1,12,1))
        i6 = np.reshape(i6[idx], (1,12,1))
        i7 = np.reshape(i7[idx], (1,12,1))
        
        test =[i1, i2, i3, i4, i5, i6, i7]
        functors = [K.function([*inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions
        layer_outs = [func([*test, 0]) for func in functors]
        
        n=0
        feat =['HR_mean', 'RR_mean', 'SBP_mean', 'SPO2_mean', 'TEMP_mean', 'avpu', 'supplemental_oxygen']
        for i in range(-17, -10):
            layer_num = i
            ax = sns.heatmap(layer_outs[layer_num][0][0], linewidths=0.05)
            plt.savefig(figs_path+key+'idx='+str(idx)+'_att_vis_layer='+str(layer_num)+'vital='+feat[n]+'.jpg', dpi=300)
            plt.show()
            n= n+1    
        
    return y, output

#%%
import seaborn as sns
def heatmap_attention(y, figs_path, idx_code, vmin, vmax, cmap):
    Z=np.array(y)
    ax = sns.heatmap(Z, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yticklabels(['HR', 'RR', 'SBP', 'SPO$_2$', 'TEMP', 'AVPU', 'Supplemental\n Oxygen'], verticalalignment='center', rotation='horizontal')
    ax.set_xticklabels(np.arange(1,13,1), rotation='horizontal', fontsize=17)
    plt.xlabel('Timesteps')
    plt.savefig(figs_path+idx_code+'attention_heatmap.png',bbox_inches='tight', dpi = 300)
    plt.show()
    
#%%
# Plot raw vitals
def plot_vitals_raw(obs, h, figs_path, t1=0, t2=0):
    vitals = obs.loc[obs.hadm_id == h]
    mpl.rcParams['font.size'] = 20
    colors = {'HR': '#2F8FC6', 'RR': '#FA8072', 'SBP':'#3CB371', 'SPO2': '#b9b9b9', 'TEMP':'gold', 
              'HGB': '#FFA500', 'WBC': '#BA55D3', 'POT':'#A0522D', 'ALB': '#2F4F4F', 'SOD': '#FF69B4', 
              'UR':'#008B8B' , 'CR': '#FF8C00', 'avpu': 'k', 'O2TH':'darkorange', 'supplemental_oxygen': 'darkorange'}
    
    v = ['HR', 'RR', 'SBP', 'SPO2', 'TEMP', 'avpu', 'O2TH']    
    eventtime = vitals.iloc[0].hrs_to_nextevent+vitals.iloc[0].hrs_from_adm
    n_event = vitals.iloc[0].next_event
    if n_event ==0:
        ec = 'k'
    else:
        ec = 'r'
    
    i=0
    #figsize = (12,10)
    fig, ax = plt.subplots(7,1, figsize=(5,10), sharex=True)
    fig.subplots_adjust(hspace=0.5)
    for var in v:
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].tick_params(axis=u'both', which=u'both',length=0)
    
        #ax[i].spines['left'].set_visible(False)
        
        nan_pos = np.argwhere(np.isnan(vitals[var].values))
        X = np.delete(vitals.hrs_from_adm.values, nan_pos)
        Y= np.delete(vitals[var].values, nan_pos)
        
        ax[i].plot(X, Y, 'o--', color=colors[var])        
        labelpad = 40
        if var == 'SPO2':
            var = 'SPO$_2$'
        elif var == 'avpu':
            var = 'AVPU'
            labelpad = 60
        elif var == 'O2TH':
            var = 'Supplemental\n Oxygen'
            labelpad = 85
        
        ax[i].set_ylabel(var, labelpad=labelpad, rotation=0)
    
        
        i+=1
    plt.axvline(eventtime, color=ec)
    
    #plt.xlabel('Hours from admission')
    #plt.tight_layout()
    if (t1==0) & (t2==0):
        plt.xlim([-1,eventtime+10])
        plt.xticks(np.arange(-1, eventtime+10, 10))
    else:
        plt.xlim([t1,t2])
        plt.xticks(np.arange(t1,t2,2))
    
    plt.xlabel('Hours from Admission',  labelpad=20)
    
    plt.savefig(figs_path+str(h)+'.png', dpi =300, bbox_inch='tight')
    plt.show()    
    
    
#%% Plot vitals scaled 
def plot_vitals_scaled(temp_set, idx_code, h,  figs_path):
    vitals = temp_set.loc[temp_set.code == idx_code]
    i=0
    fig, ax = plt.subplots(7,1, figsize=(5,10), sharex=True)
    fig.subplots_adjust(hspace=0.5)
    v = ['HR', 'RR', 'SBP', 'SPO2', 'TEMP', 'avpu', 'O2TH']    
    #eventtime = vitals.iloc[0].hrs_to_nextevent+vitals.iloc[0].hrs_from_adm

    
    colors = {'HR': '#2F8FC6', 'RR': '#FA8072', 'SBP':'#3CB371', 'SPO2': '#b9b9b9', 'TEMP':'gold', 
              'HGB': '#FFA500', 'WBC': '#BA55D3', 'POT':'#A0522D', 'ALB': '#2F4F4F', 'SOD': '#FF69B4', 
              'UR':'#008B8B' , 'CR': '#FF8C00', 'avpu': 'k', 'O2TH':'darkorange', 'supplemental_oxygen': 'darkorange'}
    for var in v:
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].tick_params(axis=u'both', which=u'both',length=0)
    
        #ax[i].spines['left'].set_visible(False)
        
        #nan_pos = np.argwhere(np.isnan(vitals[var+'_mean'].values))
       
        
        fontsize = 20
        labelpad = 25
        if var == 'SPO2':
            leg = 'SPO$_2$'
        elif var == 'avpu':
            leg = 'AVPU'
        elif var == 'O2TH':
            leg = 'Supplemental\n Oxygen'
            fontsize = 12
            labelpad = 45
            var = 'supplemental_oxygen'
        else:
            leg = var
        
        X = np.arange(0,12,1)
        Y= vitals[var+'_mean']
    
        if (var != 'avpu') & (var != 'supplemental_oxygen'):
            yerr=vitals[var+'_var'].values
            ax[i].errorbar(X, Y,  yerr=yerr, color=colors[var], capsize=3, marker='o', linestyle='--')
            ax[i].set_ylim([-1.5,1.5])
        else:
        
            ax[i].plot(X, Y, 'o--', color=colors[var])
        ax[i].set_ylabel(leg, fontsize=fontsize, labelpad=labelpad, rotation=0)
    
        
        i+=1
    #plt.axvline(eventtime, color='r')
    
    #plt.xlabel('Hours from admission')
    #plt.tight_layout()
    plt.xlim([0,12])
    
    plt.xlabel('Timesteps', fontsize=15, labelpad=20)
    plt.xticks(np.arange(0, 12, 1), np.arange(1,13,1), fontsize=20)
    plt.savefig(figs_path+str(h)+idx_code+'_scaled.png', dpi =300, bbox_inches='tight')
    plt.show()
