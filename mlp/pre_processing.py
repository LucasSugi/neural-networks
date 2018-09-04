# coding: utf-8
import pandas as pd
import numpy as np

#Read data
df = pd.read_table('wine.data',sep=',',header=None)

#Extract just the attributes
wine = df.iloc[:,1:len(df.columns)]

#Normalizing data
wine = (wine-wine.min())/(wine.max()-wine.min())

#Extract class
classes = df[0].unique()

#Generate class
tmp_class = np.zeros([len(df),len(classes)])

for i in range(len(classes)):
    for j in range(len(df)):
        if(classes[i] == df.iloc[j,0]):
            tmp_class[j,i] = 1
    
#Convert to pandas's data frame
tmp_class = pd.DataFrame(tmp_class)

#Concat with attributes
wine = pd.concat([wine,tmp_class],axis=1)

#Write file
wine.to_csv('wine_pre.data',sep=',',header=None,index=False)

#Read data
df = pd.read_table('default_features_1059_tracks.txt',sep=',',header=None)

#Extract just the attributes
tracks = df.iloc[:,0:len(df.columns)-2]

#Normalizing data
tracks = (tracks-tracks.min())/(tracks.max()-tracks.min())

#Extract class
classes = df[69].unique()

#Generate class
tmp_class = np.zeros([len(df),len(classes)])

for i in range(len(classes)):
    for j in range(len(df)):
        if(classes[i] == df.iloc[j,69]):
            tmp_class[j,i] = 1
            
#Convert to pandas's data frame
tmp_class = pd.DataFrame(tmp_class)

#Concat with attributes
tracks = pd.concat([tracks,tmp_class],axis=1)

#Write file
tracks.to_csv('default_features_1059_tracks_pre.txt',sep=',',header=None,index=False)
