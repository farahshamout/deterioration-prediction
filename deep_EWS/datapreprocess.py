import pandas as pd
import numpy as np


def fillna_median(obs):
    obs.HR = obs.HR.fillna(obs.HR.median())
    obs.SBP = obs.SBP.fillna(obs.SBP.median())
    obs.DBP = obs.DBP.fillna(obs.DBP.median())
    obs.TEMP = obs.TEMP.fillna(obs.TEMP.median())
    obs.RR = obs.RR.fillna(obs.RR.median())
    obs.SPO2 = obs.SPO2.fillna(obs.SPO2.median())
    return obs


# exclude observations recorded after an event (prev_event type is admission)
def excl_after_event(dataset): # used 
    return dataset.loc[dataset.prev_event==0]

# Exclude admissions with no observations within the last 24 hours 
def drop_adm_no24(dataset): # used 
    hadm_ids_drop = dataset.drop_duplicates(subset='hadm_id', keep='last').loc[dataset.hrs_to_nextevent>24]['hadm_id'].values
    temp_obs = dataset[~dataset.hadm_id.isin(hadm_ids_drop)]
    return temp_obs, hadm_ids_drop

# this excludes both maternity and elective admissions
def exclude_electives(dataset):
    print('Excluding elective admissions:')
    return dataset.loc[dataset.elective_admission==0]

def exclude_dayadm(dataset):
    # Exclude patients who were discharged alive AND were discharged before midnight on the day of admission
    # alive means discharge method is NOT equal to 4
    # discharged before midnight on day of admission i.e. discharge time < midnight of (admission day)
    dataset['admitdate_midnight'] = dataset['admittime'].astype(str).str[0:10] + ' 23:59:59'
    dataset['disch_b4_midn'] = dataset['dischtime'].astype(str) <= dataset['admitdate_midnight']
    dataset = dataset.drop(dataset[(dataset.death_inh != 1)&(dataset.disch_b4_midn == True)].index)
    return dataset

def summarize_df(obs): # used  
    print('Number of observations (rows): ' + str(len(obs)))
    print('Number of observations within 24 hours to an event' + str(len(obs.loc[obs.hrs_to_firstevent<=24])/len(obs)))
    
    print('Number of patient admissions:  ' + str(len(obs.drop_duplicates(subset = 'hadm_id'))))
    hadmids = obs.drop_duplicates(subset = 'hadm_id', keep = 'last')
    print('Number of patient admissions with at least 1 composite outcome' + str(len(hadmids.loc[hadmids.hrs_to_firstevent.notnull()])/len(hadmids)))
    

# Pre-process training set to illustrate performance of NEWS in the start
def preprocess_dataset(test_set):
    summarize_df(test_set)
    
    test_set = exclude_electives(test_set)
    summarize_df(test_set)
    
    # Exclude patients who were discharged before midnight on the day of admission 
    print('\nExcluding day admissions ')
    test_set  = exclude_dayadm(test_set)
    summarize_df(test_set)
    
    ## Exclude observations recorded AFTER FIRST EVENT i.e. hrs_to_firstevent is negative
    # test_set= test_set.drop(test_set[test_set.hrs_to_firstevent < 0].index)
    test_set= excl_after_event(test_set)
    summarize_df(test_set)
    
    # Drop observations with missing values
    print('Dropping observations with missing values: HR, RR, SBP, SPO2, TEMP')
    test_set = test_set.dropna(subset=['HR', 'RR', 'SBP', 'SPO2', 'TEMP'])
    summarize_df(test_set)
    
    # Pad missing AVPU and mask values
    print('Padding AVPU and masktype values')
    test_set.avpu = test_set.avpu.fillna(1)
    test_set.masktype = test_set.masktype.fillna(1)
    
    return test_set

def exclude_impossible(df):
    df = df.loc[(df.HR>=15) & (df.HR <= 250)]
    print(len(df))
    # Respiratory Rate
    df = df.loc[(df.RR>=3) & (df.RR <= 25)]
    print(len(df))
    # Temperature
    df = df.loc[(df.TEMP>=25.0) & (df.TEMP <= 45.0)]
    print(len(df))
    # Systolic blood pressure
    df = df.loc[(df.SBP>=30) & (df.SBP <= 300)]
    print(len(df))
    # oxygen saturation 
    df = df.loc[(df.SPO2>=40) & (df.SPO2 <= 100)]
    print(len(df))
    return df

def exclude_impossible_labs(df):
    # sodium
    df = df.loc[((df.SOD>=99) & (df.SOD <= 200)) | (df.SOD.isnull())]  # originally SOD < 100 eliminates many rows with values of 99.9
    print(len(df))
    # albumin
    df = df.loc[ (df.ALB <= 70) | (df.ALB.isnull())] #df.loc[(df.ALB>=10) & (df.ALB <= 70)]
    print(len(df))
    # potassium
    df = df.loc[((df.POT>=0.9) & (df.POT <= 15))|(df.POT.isnull())]
    print(len(df))
    # UR 
    df = df.loc[((df.UR>=0.4) & (df.UR <= 110))| (df.UR.isnull())] # originally 107.1 but other values are very close! 
    print(len(df))
    # CR
    df = df.loc[((df.CR>=7.9) & (df.CR <= 3000))|(df.CR.isnull())] # originally 8.8 and 2210
    print(len(df))
    return df
