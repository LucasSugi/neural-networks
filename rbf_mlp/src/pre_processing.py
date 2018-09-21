# coding: utf-8

import pandas as pd
import numpy as np

#Read table
seed = pd.read_table('dataset/seeds_dataset.txt',sep='\t',header=None)

#Extract just the attributes
attributes = seed.iloc[:,0:seed.shape[1]-1]

#Normalizing data
attributes = (attributes-attributes.min()) / (attributes.max()-attributes.min())

#Extract class
classes = seed.iloc[:,seed.shape[1]-1].unique()

#Generate class
tmp_class = np.zeros([len(seed),len(classes)])

for i in range(len(classes)):
    for j in range(len(seed)):
        if(classes[i] == seed.iloc[j,seed.shape[1]-1]):
            tmp_class[j,i] = 1
            
#Convert to pandas's data frame
tmp_class = pd.DataFrame(tmp_class)

#Concat with attributes
seed = pd.concat([attributes,tmp_class],axis=1)

#Write table
seed.to_csv('dataset/seeds_dataset_pre.txt',sep=',',header=None,index=False)
