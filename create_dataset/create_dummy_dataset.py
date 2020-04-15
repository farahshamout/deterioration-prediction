import pandas as pd
from random import seed, gauss, choice
import random 
import numpy as np 

# seed random number generator
seed(1)



# Create dummy pandas dataframe to calculate EWS
distributions = {'HR':[60,5], 'RR':[17,4], 'SBP':[100, 40], 'TEMP':[36.5,0.7],'SPO2':[95,5],'AVPU':[1,2,3,4], 'masktype':[0,1]}
cont_var = ['HR', 'RR', 'SBP', 'TEMP', 'SPO2']
cat_var =['AVPU', 'masktype']


num_patients = 10
num_observations=1000

data={}
data['hadm_id']= random.choices(list(range(0, num_patients+1)), k=num_observations)

# Sample distributions of continuous variables
for var in cont_var:
  data[var] = [np.round(gauss(distributions[var][0],distributions[var][1]),2) for _ in range(num_observations)]


# Sample categorical variables 
for var in cat_var:
  data[var]=random.choices(distributions[var],k= num_observations)

df = pd.DataFrame(data).sort_values('hadm_id').reset_index(drop=True)

df.to_csv('dummy_obs.csv')
