#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Standard library imports
import os

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import dill
import pandas as pd

# Local application imports
from smile import helper

# Settings
pickle_dir = r'D:\smile data\saved_populations_11_large'
csv_dir = r'D:\smile data\saved_populations_11_large_csv'


# In[2]:


def load_from_file(filename):
    with open(filename, 'rb') as f:
        return dill.load(f)


# In[3]: Generated

for i in range(97,100):
    poplists = load_from_file(pickle_dir+'\worddoc_poplists_'+str(i)+'.pik')
    for j in range(10):
        for k,slope in enumerate(['1','2','3']):
            for l,err in enumerate(['03', '05']):
                csv_inner_dir = r'\realistic_'+slope+'_'+err
                csv_name = csv_dir+csv_inner_dir+csv_inner_dir+'_population_'+str(10*i+j)+'.csv'
                csv_df = pd.read_csv(csv_name, index_col='observation')
                pickle_df = poplists[k,l][j].to_dataframe()
                pickle_df = pickle_df.astype({'person': np.int64, 'day': np.int64})
                pickle_df = pickle_df.astype({'visual': np.float64, 'symptom_noerror': np.float64, 'symptom': np.float64})

                pd.testing.assert_frame_equal(csv_df, pickle_df)

    print(f"{i}/100")



# In[3]: Filtered
'''
for i in range(100):
    poplists = load_from_file(pickle_dir+'\worddoc_filtered_poplists_'+str(i)+'.pik')
    for j in range(10):
        for k,slope in enumerate(['1','2','3']):
            for l,err in enumerate(['03', '05']):
                csv_inner_dir = r'\realistic_'+slope+'_'+err
                csv_name = csv_dir+csv_inner_dir+csv_inner_dir+'_population_'+str(10*i+j)+'_filtered.csv'
                csv_df = pd.read_csv(csv_name, index_col='observation')
                pickle_df = poplists[k,l][j].to_dataframe()
                pickle_df = pickle_df.astype({'person': np.int64, 'day': np.int64})
                pickle_df = pickle_df.astype({'visual': np.float64, 'symptom_noerror': np.float64, 'symptom': np.float64})

                pd.testing.assert_frame_equal(csv_df, pickle_df)

    print(f"{i}/100")
'''


# In[3]: Sampled

for i in range(100):
    poplists = load_from_file(pickle_dir+'\worddoc_sampled_poplists_'+str(i)+'.pik')
    for j in range(10):
        for k,slope in enumerate(['1','2','3']):
            for l,err in enumerate(['03', '05']):
                for m,method in enumerate(['traditional', 'realistic']):
                    csv_inner_dir = r'\realistic_'+slope+'_'+err
                    csv_name = csv_dir+csv_inner_dir+csv_inner_dir+'_population_'+str(10*i+j)+'_sampled_'+method+'.csv'
                    csv_df = pd.read_csv(csv_name, index_col='observation')
                    pickle_df = poplists[k,l,m][j].to_dataframe()
                    pickle_df = pickle_df.astype({'person': np.int64, 'day': np.int64})
                    pickle_df = pickle_df.astype({'visual': np.float64, 'symptom_noerror': np.float64, 'symptom': np.float64})

                    pd.testing.assert_frame_equal(csv_df, pickle_df)

    print(f"{i}/100")

