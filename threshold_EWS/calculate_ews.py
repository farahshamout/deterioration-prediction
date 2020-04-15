import pandas as pd 
import ews_thresholds as ews 

# Import dummy dataset
obs = pd.read_csv("../dataset/dummy_obs.csv")

# Activate early warning score thresholds 
news1 =pd.DataFrame(ews.NEWS1().thresh, columns=['VAR', 'MIN', 'MAX', 'SCORE'])
cews = pd.DataFrame(ews.CEWS().thresh, columns=['VAR', 'MIN', 'MAX', 'SCORE'])
mcews_mp = pd.DataFrame(ews.MCEWS_MP().thresh, columns=['VAR', 'MIN', 'MAX', 'SCORE'])


# Calculate EWS and save new dataframe 
obs['NEWS'] = ews.calculate_ews(news1, obs, 'NEWS')
obs['MCEWS'] = ews.calculate_ews(mcews_mp, obs, 'MCEWS')
obs['CEWS'] = ews.calculate_ews(cews, obs, 'CEWS')

obs.to_csv("../dataset/dummy_obs_ews.csv")
