# Standard library imports
import os

# Third party imports
import numpy as np
import dill
import pandas as pd

# Local application imports
#none




# Settings
seed = 3 # chosen by fair dice roll. guaranteed to be random. https://xkcd.com/221/
np.random.seed(seed)
np.set_printoptions(edgeitems=30, linewidth=100000)
pickle_pops_dir = r'C:\Users\zachv\Desktop\smile_desk\simulating_randomness_test_11'
pickle_csv_dir = r'C:\Users\zachv\Desktop\smile_desk\simulating_randomness_test_11_csv'




def load_from_file(filename):
    with open(filename, 'rb') as f:
        return dill.load(f)
def dump_to_csv_file(df, filename, filesuffix='.csv', 
                     dirname=None, create_newdir=False, avoid_overwrite=True):
    #build path
    if dirname is not None: 
        if not os.path.isdir(dirname):
            if create_newdir:
                os.makedirs(dirname)
                print(f"Directory {dirname} was created")
            else:
                raise OSError(f"Directory {dirname} doesn't exist and needs to be created")
        filename = os.path.join(dirname, filename+filesuffix)
    else:
        filename = filename+filesuffix
    #check if will overwrite
    if os.path.isfile(filename) and avoid_overwrite:
        raise OSError(f"File {filename} already exists and would be overwritten")
    else:
        df.to_csv(filename)




#parameters
npersons=10 #per population
npops=5 #across all files
slope_options = (1, 2)
error_options = (0.3, 0.5)
method_options = ("traditional", "realistic")

def options_to_string(slope, error):
    return f"{slope}_{error}".replace(".","")

npops_per_sim = 2
nsims = npops // npops_per_sim
npops_remainder = npops % npops_per_sim
if npops_remainder > 0: nsims += 1

for n in range(nsims):
    #Full
    poplists = load_from_file(pickle_pops_dir+"\poplists_"+str(n)+".pik")
    for i,j in np.ndindex(poplists.shape):
        foldername = f"realistic_"+options_to_string(slope_options[i], error_options[j])
        poplist = poplists[i,j]
        for l in range(len(poplist)):
            pop = poplist[l]
            dump_to_csv_file(pop.to_dataframe(), foldername+f"_population_{n*npops_per_sim+l}", filesuffix='.csv', 
                             dirname=os.path.join(pickle_csv_dir, foldername), create_newdir=True, avoid_overwrite=True)

    #Filtered
    poplists = load_from_file(pickle_pops_dir+"\filtered_poplists_"+str(n)+".pik")
    for i,j in np.ndindex(poplists.shape):
        foldername = f"realistic_"+options_to_string(slope_options[i], error_options[j])
        poplist = poplists[i,j]
        for l in range(len(poplist)):
            pop = poplist[l]
            dump_to_csv_file(pop.to_dataframe(), foldername+f"_population_{n*npops_per_sim+l}_filtered", filesuffix='.csv', 
                             dirname=os.path.join(pickle_csv_dir, foldername), create_newdir=True, avoid_overwrite=True)

    #Sampled
    poplists = load_from_file(pickle_pops_dir+"\sampled_poplists_"+str(n)+".pik")
    for i,j, k in np.ndindex(poplists.shape):
        foldername = f"realistic_"+options_to_string(slope_options[i], error_options[j])
        poplist = poplists[i,j,k]
        for l in range(len(poplist)):
            pop = poplist[l]
            df = pop.to_dataframe()
            #Basic
            dump_to_csv_file(df, foldername+f"_population_{n*npops_per_sim+l}_sampled_{method_options[k]}", filesuffix='.csv', 
                             dirname=os.path.join(pickle_csv_dir, foldername), create_newdir=True, avoid_overwrite=True)
            #Days only
            dfscoreless = df[['person', 'day']] #remove scores
            dflists = dfscoreless.groupby('person').agg(list) #aggregate each person's days into lists
            dfdays = pd.DataFrame(dflists['day'].to_list()).set_index(dflists.index) #split lists back into columns
            dump_to_csv_file(dfdays, foldername+f"_population_{n*npops_per_sim+l}_sampled_{method_options[k]}_daysonly", filesuffix='.csv',
                             dirname=os.path.join(pickle_csv_dir, foldername), create_newdir=True, avoid_overwrite=True)

            
# use python -u to_csv.py 2>&1 | tee to_csv_out.txt