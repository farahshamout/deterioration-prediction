## Detection of In-hospital Deterioration on Hospital Wards



This repository includes code from Shamout's DPhil titled "Machine Learning for the Detection of Clinical Deterioration on Hospital Wards". 



*Note that this repository will be constantly updated and is not final yet...*



The work involves three types of models:

- Threshold-based Early Warning Score
- Deep Interpretable Early Warning Score
- Multi-modal Early Warning Score 



Steps to run the code:

- create_dummy_dataset.py - this script generates a dummy dataset of the most routinely collected vital signs (HR, SBP, SPO2, TEMP, AVPU, & masktype). masktype indicates the provision of supplemental oxygen
- navigate to the score of interest and run its respective scripts 

calculate_ews.py - this scripts calculates the threshold based EWS scores for the dummy dataset based on the score of interest, e.g. NEWS, MCEWS, or CEWS 



## Acknowledgements 

Thanks to my supervisors Prof. David Clifton and Dr. Tingting Zhu and clinical collaborator Dr. Peter Watkinson.



