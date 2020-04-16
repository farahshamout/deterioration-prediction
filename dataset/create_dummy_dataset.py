import pandas as pd
from random import seed, gauss, choice
import random 
import numpy as np 
from utils import random_date


# seed random number generator
seed(1)

# Create dummy pandas dataframe to calculate EWS
distributions = {'HR':[60,5], 'RR':[17,4], 'SBP':[100, 40], 'TEMP':[36.5,0.7],'SPO2':[95,5],'avpu':[1,2,3,4], 'masktype':[0,1]}
cont_var = ['HR', 'RR', 'SBP', 'TEMP', 'SPO2']
cat_var =['avpu', 'masktype']


num_patients = 30
num_observations=1000
hadm_ids = list(range(0, num_patients))

sex = ['F','M']

# Sample random admission, discharge, demographics, and adverse event times 
encounter_data = pd.DataFrame()
t1 = '2017-01-01 00:00:00'
t2 = '2019-12-31 23:59:59'
admittimes = [random_date(t1, t2, random.random()) for _ in range(num_patients)]
dischtimes = [random_date(x, t2, random.random()) for x in admittimes]
eventtimes = [random_date(x,y, random.random()) for x,y in zip(admittimes, dischtimes)]
age = random.choices(list(range(16,90)), k=num_patients)
sex = random.choices(sex, k=num_patients)
encounter_data=pd.DataFrame({'hadm_id': hadm_ids,'admittime': admittimes, 'dischtime': dischtimes, 'age': age, 'sex': sex, 'eventtime': eventtimes})
encounter_data.loc[encounter_data.sample(frac=0.7).index, 'eventtime'] = pd.np.nan
encounter_data['next_event']=np.where(encounter_data.eventtime.isna(), 0, random.choices([1,2,3],k=1)[0])


# Sample observations dataframe
data={}
data['hadm_id']= random.choices(hadm_ids, k=num_observations)

# Sample distributions of continuous variables
for var in cont_var:
  data[var] = [np.round(gauss(distributions[var][0],distributions[var][1]),2) for _ in range(num_observations)]

# Sample categorical variables 
for var in cat_var:
  data[var]=random.choices(distributions[var],k= num_observations)
df = pd.DataFrame(data).reset_index(drop=True)

# Merge with the encounter information
df = df.merge(encounter_data, how='left', on='hadm_id')
df['charttime'] = [random_date(x,y,random.random()) for x, y in zip(df.admittime.values, df.dischtime.values)]
df[['eventtime', 'charttime']] = df[['eventtime', 'charttime']].apply(pd.to_datetime, errors='coerce')
df['hrs_to_firstevent'] = (df.eventtime - df.charttime).astype('timedelta64[h]')
df['prev_event']=np.where(df.hrs_to_firstevent<0, 1,0)
df.sort_values(['hadm_id', 'charttime']).to_csv('dummy_obs.csv', index=False)
