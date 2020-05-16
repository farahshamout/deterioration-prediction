import pandas as pd
import numpy as np


"""
This file contains functions to calculate the total EWS based on pre-defined alerting thresholds.
"""



def calculate_ews(EWS, df, label='Score', name='', train_cols=[]):
    """ Adopted from Marco Pimentel:
        function to calculate EWS score
     EWS:  dataframe with components of the score and respective cutoffs
           VAR: string with variable code, e.g., 'HR'
           MIN: minimum cutoff value of an interval for a given score
           MAX: maximum cutoff value of an interval for a given score
           SCORE: value of the score for the given cutoff interval
     df:   dataframe with data components to be scored
     label: string with the name of the score (optional)
    """
    if label is None:
        label = 'Score'

    # get variables to be scored
    if not train_cols:
        if name is 'CEWS':
            train_cols = ['HR', 'RR', 'TEMP', 'SPO2', 'SBP']
        else:
            train_cols = ['HR', 'RR', 'TEMP', 'SPO2', 'SBP', 'masktype', 'avpu']
    temp_df = pd.DataFrame()
    # loop over each component, and map score
    for x in train_cols:
        temp_df[label + '_' + x] = pd.cut(df[x],
                                  bins=[-1] + EWS[['MIN', 'MAX']][EWS.VAR == x].stack()[1::2].tolist(), 
                                     labels=EWS.SCORE[EWS.VAR == x].tolist(), right=True).str.replace('X', '').astype('float')

    # define columns to be added
    score_cols = [col for col in temp_df if col.startswith(label + '_')]
    
    # calculate final score (don't forget to use the absolute value)
    df[label] = np.absolute(temp_df[score_cols]).sum(axis=1)
    
    if len(train_cols)==1:
        return df
    else:
        #return df 
        return df[[label]]


def calculate_asews( dataset_eval, asews, train_cols=[]):
    """
     Calculate the age and sex based EWS (ASEWS)
    """
    asews_obs = pd.DataFrame()
    M= ['F', 'M']
    for s in range(0, len(M)):
        test_set_g = dataset_eval.loc[dataset_eval['gender']==M[s]]
        min_age = int(test_set_g.age.min())
        max_age = int(test_set_g.age.max())
        for i in range(min_age, max_age+1):
            sub_test = test_set_g.loc[test_set_g.age==i]
            sub_ews = asews[M[s]][i]
            scored_obs = calculate_ews(sub_ews, sub_test[['HR', 'RR', 'TEMP', 'SPO2', 'SBP', 'avpu', 'masktype', 'hrs_to_firstevent', 'age', 'gender']], 'EWS')
            if len(asews_obs) != 0:
                asews_obs=asews_obs.append(scored_obs, ignore_index=False)
            else: 
                asews_obs = scored_obs
    asews_obs['index'] = asews_obs.index
    asews_obs = asews_obs.sort_values(by=['index'])
    return asews_obs[['EWS']]   
 #return asews_obs[['hrs_to_firstevent', 'EWS', 'age', 'gender']]


def calculate_aews(asews, dataset_eval, name):
    """
    Calculate the Age based EWS
    """

    asews_obs = pd.DataFrame()
    min_age = int(dataset_eval.age.min())
    max_age = int(dataset_eval.age.max())
    max_ews = max(list(asews.keys()))
    for i in range(min_age, max_age+1):
        sub_test = dataset_eval.loc[dataset_eval.age==i]
        if i > max_ews:
            sub_ews=asews[max_ews]
        else:
            sub_ews = asews[i]
        scored_obs = calculate_ews(sub_ews, sub_test[['HR', 'RR', 'TEMP', 'SPO2', 'SBP', 'avpu', 'masktype', 'hrs_to_firstevent', 'age', 'gender']], 'EWS')
        if len(asews_obs) != 0:
            asews_obs=asews_obs.append(scored_obs, ignore_index=True)
        else: 
            asews_obs = scored_obs
    return asews_obs[['EWS']]
    #return asews_obs[['hrs_to_firstevent', 'EWS', 'age', 'gender']]


## classes of ews thresholds
class NEWS1(object):
    """
    Thresholds of the NEWS score 
    """
    thresh =[]
    
    def __init__(self):
        self.thresh=[['HR', -1, 40, '3'],
             ['HR', 41, 50, '1'],
             ['HR', 51, 90, '0'],
             ['HR', 91, 110, '1X'],
             ['HR', 111, 130, '2'],
             ['HR', 131, 250, '3X'],
             
            ['RR', -1, 8, '3'], 
            ['RR', 9, 11, '1'], 
            ['RR', 12, 20, '0'],
            ['RR', 21, 24, '2'],
            ['RR', 25, 65, '3X'],
            
            ['SPO2', -1, 91, '3'],
            ['SPO2', 92, 93, '2'], 
            ['SPO2', 94, 95, '1'], 
            ['SPO2', 96, 101, '0'], 
            
            ['masktype', -1, 1, '0'], 
            ['masktype', 2, 23, '2'],
            
            ['SBP', -1, 90, '3'],
            ['SBP', 91, 100, '2'], 
            ['SBP', 101, 110, '1'],
            ['SBP', 111, 219, '0'],
            ['SBP', 220, 300, '3X'],
            
             ['avpu', -1, 1, '0'],
             ['avpu', 2, 4, '3'],
             
             ['TEMP', -1, 35.0, '3'],
             ['TEMP', 35.1, 36.0, '1'],
             ['TEMP', 36.1, 38.0, '0'],
            ['TEMP', 38.1, 39.0, '1X'],
             ['TEMP', 39.1, 50.0, '2']
            ]

class CEWS(object):
    """
    Thresholds of the Centile-based EWS (CEWS): https://www.ncbi.nlm.nih.gov/pubmed/21482011
    """
    thresh=[]
    
    def __init__(self):
        self.thresh=[
            ['HR', -1, 50, '3'],
            ['HR', 51, 58, '2'],
            ['HR', 59, 63, '1'],
            ['HR', 64, 104, '0'],
            ['HR', 105, 112, '1X'],
            ['HR', 113, 127, '2X'],
            ['HR', 128, 250, '3X'],
            
            ['RR', -1, 7, '3'],
            ['RR', 8, 10, '2'],
            ['RR', 11, 13, '1'],
            ['RR', 14, 25, '0'],
            ['RR', 26, 28, '1X'],
            ['RR', 29, 33, '2X'],
            ['RR', 34, 61, '3X'],
            
            ['TEMP',-1, 35.4, '3'],
            ['TEMP', 35.5, 35.9, '1'],
            ['TEMP', 36.0, 37.3, '0'],
            ['TEMP', 37.4, 38.3, '1X'],
            ['TEMP', 38.4, 50, '3X'],
            
            ['SBP', -1, 85, '3'],
            ['SBP', 86, 96, '2'],
            ['SBP', 97, 101, '1'],
            ['SBP', 102, 154, '0'],
            ['SBP', 155, 164, '1X'],
            ['SBP', 165, 184, '2X'],
            ['SBP', 185, 300, '3X'],
            
            
            ['SPO2', -1, 84, '3'],
            ['SPO2', 85, 90, '2'],
            ['SPO2', 91, 93, '1'],
            ['SPO2', 94, 101, '0'],
            
            ['avpu',-1, 1, '0'],
            ['avpu', 2, 2, '1'],
            ['avpu', 3, 4, '3']
            
            
        ]

class MCEWS_MP(object):
    """
    Thresholds of the modified centile-based EWS: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6062656/
    """

    thresh=[]
    
    def __init__(self):
        self.thresh=[
            ['HR', -1, 42, '3'],
            ['HR', 43, 49, '2'],
            ['HR', 50, 53, '1'],
            ['HR', 54, 104, '0'],
            ['HR', 105, 112, '1X'],
            ['HR', 113, 127, '2X'],
            ['HR', 128, 250, '3X'],
            
            ['RR', -1, 7, '3'],
            ['RR', 8, 11, '2'], 
            ['RR', 11, 12, '1'],
            ['RR', 13, 21, '0'],
            ['RR', 22, 23, '1X'],
            ['RR', 24, 28, '2X'],
            ['RR', 29, 61, '3X'],
            
            ['TEMP', -1, 35.4, '3'],
            ['TEMP', 35.5, 35.9, '1'],
            ['TEMP', 36.0, 37.3, '0'],
            ['TEMP', 37.4, 38.3, '1X'],
            ['TEMP', 38.4, 50, '3X'],
            
            ['SBP', -1, 83, '3'],
            ['SBP', 84, 90, '2'],
            ['SBP', 91, 100, '1'],
            ['SBP', 101, 157, '0'],
            ['SBP', 158, 167, '1X'],
            ['SBP', 168, 184, '2X'],
            ['SBP', 185, 300, '3X'],
            
            
            ['SPO2', -1, 84, '3'],
            ['SPO2', 85, 90, '2'],
            ['SPO2', 91, 93, '1'],
            ['SPO2', 94, 101, '0'],
            
            ['masktype', -1, 1, '0'], 
            ['masktype', 2, 23, '2'],
            
            ['avpu', -1, 1, '0'],
            ['avpu', 2, 2, '1'],
            ['avpu', 3, 4, '3']
        ]





class LDTEWS_M(object):
    thresh = []
    
    def __init__(self):
        self.thresh = [['HGB', 0, 11.1, '2'],
                        ['HGB', 11.2, 12.8, '1'], 
                        ['HGB', 12.9, 5000, '0'], 
                        
                        ['WBC', -1, 9.3, '0'], 
                        ['WBC', 9.4, 16.6, '1'], 
                        ['WBC', 16.7, 5000, '2'], 
                        
                        ['UR', -1, 9.4, '0'],
                        ['UR', 9.5, 13.7, '1'], 
                        ['UR', 13.8, 5000, '3'],
                        
                        ['CR', -1, 114, '0'], 
                        ['CR', 115, 179, '1'], 
                        ['CR', 180, 5000, '2'], 
                        
                        ['SOD', -1, 132, '2'], 
                        ['SOD', 133, 140, '0'], 
                        ['SOD', 141, 5000, '1'],
                        
                        ['POT', -1, 3.7, '1'], 
                        ['POT', 3.8, 4.4, '0'],
                        ['POT', 4.5, 4.7, '1X'],
                        ['POT', 4.8, 5000, '2'], 
                        
                        ['ALB', -1, 30, '2'],
                        ['ALB', 31, 34, '1'],
                        ['ALB', 35, 5000, '0']
                ]


class LDTEWS_F(object):
    thresh = []
    
    def __init__(self):
        self.thresh = [['HGB', 0, 12.0, '1'],
                        ['HGB', 12.1, 14.8, '0'], 
                        ['HGB', 14.9, 5000, '1X'], 
                        
                        ['WBC', -1, 12.6, '0'], 
                        ['WBC', 12.7, 14.8, '1'], 
                        ['WBC', 14.9, 5000, '2'], 
                        
                        ['UR', -1, 8.4, '0'],
                        ['UR', 8.5, 13.8, '1'], 
                        ['UR', 13.9, 5000, '3'],
                        
                        ['CR', -1, 91, '0'], 
                        ['CR', 92, 157, '1'], 
                        ['CR', 158, 5000, '2'], 
                        
                        ['SOD', -1, 134, '2'], 
                        ['SOD', 135, 140, '0'], 
                        ['SOD', 141, 5000, '1'],
                        
                        ['POT', -1, 3.3, '1'], 
                        ['POT', 3.4, 4.5, '0'],
                        ['POT', 4.6, 5000, '1X'], 
                        
                        ['ALB', -1, 28, '2'],
                        ['ALB', 29, 34, '1'],
                        ['ALB', 35, 5000, '0']
                ]        
 
