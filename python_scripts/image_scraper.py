### imports

from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

import urllib.request as ur

import os


### get all files in a directory


# Point this path to where you store the 'image_csv' directory locally
mypath = '/Users/ggarbagnati/ds/metis/metisgh/sf17_ds5/local/Projects/05-Kojak/image_csv/'

# making a list of all .csv files in the 'image_csv' dir
image_url_csvs = [f for f in listdir(mypath) if isfile(join(mypath, f))]
image_url_csvs = [f for f in image_url_csvs if '.csv' in f]


### Scraping list of dfs

def impute_disease(x):
    if pd.isnull(x):
        return 'healthy'
    elif len(x) < 1:
        return 'healthy'
    else:
        return x

dfs = []

for csv in image_url_csvs:
    
    filename = mypath + csv
    #print(filename)
    df = pd.read_csv(filename, header = 3)

    df['Disease common name'] = df['Disease common name'].apply(lambda x: impute_disease(x))
    
    df['scraped'] = 0
    
    dfs.append(df)

#print(len(dfs))



### scraping each image

# I set this limit to for when I was first testing the script in a ipython
#  notebook so it's not important now
dl_limit = 1000000

j = 0

for df in dfs:

    for i, url in enumerate(df['url']):

        if j >= dl_limit:
            break

        if df['scraped'][i] == 1:
            continue
        
        #print(i, df['Crop common name'][i], j, url)

        directory = df['Crop common name'][i]
        if '(' in directory:
            directory = directory[:directory.find('(')-1]
        directory = directory.replace(' ', '_')
        directory = 'images/' + directory
        sub_dir_name = df['Disease common name'][i].replace(' ', '_')

        # checks to see if directories exist for each image category, and
        #  creates the directory if does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        sub_dir = directory + '/' + sub_dir_name
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        extension_i = url.rfind('.') 
        extension = url[extension_i:extension_i+4]

        filenum = '%05d' % (i,)
        filename = directory[6:] + '-' + sub_dir_name + '-' + filenum + extension

        #print(filename)
        
        filepath = sub_dir + '/' + filename

        #print(filepath)

        try:
            ur.urlretrieve(url, filepath)

            df.set_value(i,'scraped',1)
        
        except ContentTooShortError:
            pass
        
        j += 1

        
