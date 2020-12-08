"""
Created on Sun Dec  2 14:25:36 2018
Updated April 2020 

"""

import os
#path = 'C:/Users/ball4624/Desktop/PhD-research-code/Year 2 (GP-DL)/'
#os.chdir(path)

from multiprocessing import Pool
import pandas as pd
import GPy
from plot_vitals_GPS_2 import  plot_GP #, plot_GP_point_2_rmse
from time_utils import timing_decorator
import numpy as np
import datapreprocess as datapp
import gp_dl_funcs as gpdl
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")


# disable warnings 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

vital_signs = ['HR', 'RR', 'TEMP', 'SBP', 'SPO2']
vital_signs_2 = ['HR', 'RR', 'TEMP', 'SBP', 'SPO2', 'avpu', 'supplemental_oxygen']


# Set paths 
datasets_path = '/Users/farahshamout/Desktop/deterioration-prediction/dataset/'
figs_path = '/Users/farahshamout/Desktop/deterioration-prediction/deep_EWS/figures/'
raw_obs = 'dummy_obs.csv'


def upload_dataset(obs_path):
    # Get observation set and clean
    obs =  pd.read_csv(datasets_path+obs_path)
    obs = obs.loc[:, ~obs.columns.duplicated()]
    obs['hrs_to_firstevent'] = np.nan
    obs['hrs_to_firstevent']= np.where(obs['next_event']!=0, obs['hrs_to_nextevent'], obs['hrs_to_firstevent'] )
    obs['label']= obs.hrs_to_firstevent<=24
    
    # Drop observations with no patient admission (duplicates)
    obs = obs.dropna(subset=['hadm_id'])
    
    
    # Split the data based on time 
    t1 = '2013-07-01 00:00:00'
    t2= '2017-05-31 23:59:59'
    t3= '2017-06-01 00:00:00'
    t4= '2017-10-31 23:59:59'
    t5= '2017-11-01 00:00:00'
    t6 = '2018-03-31 23:59:59'
    
    train_set = obs.loc[(obs.admittime.astype(str)>=t1)&(obs.admittime.astype(str)<=t2)]
    print('Training set step 1')
    datapp.summarize_df(train_set)
    print(len(train_set)/len(obs))
    
    val_set = obs.loc[(obs.admittime.astype(str)>=t3)&(obs.admittime.astype(str)<=t4)]
    print('Validation set step 1')
    datapp.summarize_df(val_set)
    print(len(val_set)/len(obs))
    
    test_set= obs.loc[(obs.admittime.astype(str)>=t5)&(obs.admittime.astype(str)<=t6)]
    print('Testing set step 1')
    datapp.summarize_df(test_set)
    print(len(test_set)/len(obs))
    # Train set pre-processing --> drop obs after first event
    train_set = datapp.excl_after_event(train_set)
    print('Training set step 2')
    datapp.summarize_df(train_set)
    
    # Test set and validation set pre-processing --> as in ASEWS 
    val_set, _ = datapp.drop_adm_no24(datapp.preprocess_dataset(val_set))
    test_set, _ = datapp.drop_adm_no24(datapp.preprocess_dataset(test_set))
    vs = ['HR', 'RR', 'TEMP', 'SBP', 'SPO2', 'avpu', 'masktype']
    # code points 
    train_set = gpdl.number_points(train_set)
    val_set = gpdl.number_points(val_set)
    test_set = gpdl.number_points(test_set)
    
    train_set = gpdl.code_points(train_set)
    val_set = gpdl.code_points(val_set)
    test_set = gpdl.code_points(test_set)
    
    train_set['point'] = train_set.point.map(int)
    val_set['point'] = val_set.point.map(int)
    test_set['point'] = test_set.point.map(int)
    
    train_set['supplemental_oxygen']=(train_set.masktype > 1).astype(int)
    val_set['supplemental_oxygen']= (val_set.masktype > 1).astype(int)
    test_set['supplemental_oxygen']=(test_set.masktype > 1).astype(int)
    # Match abnormal obs with normal obs per each data set
    max_abn_tr = len(train_set.loc[train_set.hrs_to_firstevent<=24])  
    
    balanced_tr = pd.concat([train_set.loc[train_set.next_event==0].sample(n=max_abn_tr) , train_set.loc[train_set.hrs_to_firstevent<=24]])    
    
    max_abn_tst = len(test_set.loc[test_set.hrs_to_firstevent<=24])  
    balanced_tst=  pd.concat([test_set.loc[test_set.next_event==0].sample(n=max_abn_tst) , test_set.loc[test_set.hrs_to_firstevent<=24]])
    
    max_abn_val =len(val_set.loc[val_set.hrs_to_firstevent<=24])  
    balanced_val =  pd.concat([val_set.loc[val_set.next_event==0].sample(n=max_abn_val) , val_set.loc[val_set.hrs_to_firstevent<=24]])
    return train_set, val_set, test_set, balanced_tr, balanced_tst, balanced_val


train_set, val_set, test_set, balanced_tr, balanced_tst, balanced_val  = upload_dataset(raw_obs)


# store
train_set.to_pickle(datasets_path+'train_set_gp.pkl')
val_set.to_pickle(datasets_path+'val_set_gp.pkl')
test_set.to_pickle(datasets_path+'test_set_gp.pkl')
balanced_tr.to_pickle(datasets_path+'bl_train_set.pkl')
balanced_val.to_pickle(datasets_path+'bl_val_set.pkl')
balanced_tst.to_pickle(datasets_path+'bl_test_set.pkl')

#  load
'''train_set= pd.read_pickle(datasets_path+'train_set_gpdl_Jan6.pkl')
val_set= pd.read_pickle(datasets_path+'val_set_gpdl_Jan6.pkl')
test_set = pd.read_pickle(datasets_path+'test_set_gpdl_Jan6.pkl')
balanced_tr = pd.read_pickle(datasets_path+'bl_train_set_gpdl_Jan6.pkl')
balanced_val=pd.read_pickle(datasets_path+'bl_val_set_gpdl_Jan6.pkl')
balanced_tst=pd.read_pickle(datasets_path+'bl_test_set_gpdl_Jan6.pkl')
mean_v_age = pd.read_pickle(datasets_path+'mean_vitals_Jan6.pkl')'''


mean_v_age = train_set[['HR', 'SBP', 'TEMP', 'SPO2', 'RR', 'avpu', 'masktype']].mean()
mean_v_age['avpu'] = np.round(mean_v_age.avpu,0)
mean_v_age['masktype'] = np.round(mean_v_age.masktype,0)


train_set['point'] = 0
train_set['code'] = train_set['hadm_id']


# set up parameters for gaussian process regression
n = 48
s = int(n/1)
vitals_priors = {'HR':{'lengthscale':GPy.priors.LogGaussian(1.5, 0.1), 'noise': GPy.priors.LogGaussian(1.5, 0.1), 'variance':GPy.priors.LogGaussian(3.5, 0.1), 'plot_ylims':[30,160] },
                'RR': {'lengthscale':GPy.priors.LogGaussian(1.5, 0.1), 'noise': GPy.priors.LogGaussian(0, 0.1), 'variance':GPy.priors.LogGaussian(1.5, 0.1), 'plot_ylims':[5,45] },
                'TEMP': {'lengthscale':GPy.priors.LogGaussian(1.5, 0.1), 'noise': GPy.priors.LogGaussian(0, 20), 'variance':GPy.priors.LogGaussian(0.5, 0.1), 'plot_ylims':[33,43] },
                'SBP':{'lengthscale':GPy.priors.LogGaussian(1.0, 0.1), 'noise': GPy.priors.LogGaussian(1.5, 0.1), 'variance':GPy.priors.LogGaussian(3.5, 0.1), 'plot_ylims':[40,180] },
                'DBP': {'lengthscale':GPy.priors.LogGaussian(1.0, 0.1), 'noise': GPy.priors.LogGaussian(1.5, 0.1), 'variance':GPy.priors.LogGaussian(3.5, 0.1), 'plot_ylims':[10,175] },
                'SPO2': {'lengthscale':GPy.priors.LogGaussian(1.5, 0.1), 'noise': GPy.priors.LogGaussian(1.5, 0.1), 'variance':GPy.priors.LogGaussian(3.5, 0.1), 'plot_ylims':[40,120] }} 


store_cols = [v+'_mean' for v in vital_signs] + [v+'_var' for v in vital_signs] + ['hadm_id'] + ['point']


@timing_decorator


def my_pool(x, pool_func, func):
    '''x is a list of inputs to be processed through parallelization
       pool_func is an initiation of Pool(num_p)
       func is the function to parallelize'''
    '''pdb.set_trace()'''
    r=pool_func.starmap(func, x) # starmap takes a list of inputs with multiple arguments
    pool_func.close()
    pool_func.join()
    return r


if __name__ == '__main__':
    __spec__ = None
    
    data_id = input('Choose which dataset to model (training, testing, or validation:')
    date = input('Please input date in the format of monDD (e.g.JanXX):')
    if data_id == 'training':
        dataset = train_set
        or_dataset = train_set
        descrip= 'train_set_sampled_GP_'+date+'.pkl'
    elif data_id =='testing':
        dataset = test_set
        or_dataset = test_set
        descrip= 'test_set_sampled_GP_'+date+'.pkl'
    elif data_id == 'validation':
        dataset = balanced_val
        or_dataset=val_set 
        descrip= 'val_set_sampled_GP_'+date+'.pkl'
    else:
        data_id = input('Error! Please choose training, testing, or validation')
   
            
    hadmids = dataset.drop_duplicates(subset=('hadm_id')).hadm_id.values
    num_p = 2
    print("Modelling GPR for %d unique patient admissions using %d processors..." % (len(hadmids), num_p) )

    results = []
   
    l2= [[i, dataset.loc[dataset.hadm_id==i], or_dataset.loc[or_dataset.hadm_id==i], vital_signs_mimic, n, figs_path, vitals_priors,s, store_cols,mean_v_age] for i in hadmids]
    
    results = my_pool(l2, Pool(num_p), plot_GP)
    #descrip='rmse_'+descrip
    
    #results = my_pool(l2, Pool(num_p), plot_GP_point_2)
    descrip= str(start)+'_'+str(end) + '_'+ descrip
   
    del l2, hadmids
    
    #df = pd.concat(results) 
    df1 = pd.concat([i[0] for i in results]) 
    df2 = pd.concat([i[1] for i in results])
    df1.to_pickle(datasets_path+'/GP/'+descrip)
    df2.to_pickle(datasets_path+'/GP/'+'rmse_'+descrip)

