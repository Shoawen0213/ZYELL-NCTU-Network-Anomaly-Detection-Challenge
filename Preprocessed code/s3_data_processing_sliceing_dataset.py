#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import random
import numpy as np
import pandas as pd
from sklearn.externals import joblib


# In[2]:


print("now in...")
now_path = os.getcwd()
# file_path = os.getcwd()+ '/testing_dataset'
print(now_path)


# In[3]:


file_path = '/home/shaowen0213/NAD/Time&&IP_in_int(26col).csv'
train_data = pd.read_csv(file_path)
train_data     #9241463 rows × 26 columns


# In[4]:


unique, counts = np.unique(train_data['label'], return_counts=True)
print("xgbc_predict contains:{}".format(dict(zip(unique, counts))))


# In[5]:


train_Normal = train_data.loc[train_data['label'].str.contains('Normal')]
print("Total numbers of Normal : {}".format(len(train_Normal)))

train_DOS_smurf = train_data.loc[train_data['label'].str.contains('DOS-smurf')]
print("Total numbers of DOS-smurf : {}".format(len(train_DOS_smurf)))

train_Probing_Ipsweep = train_data.loc[train_data['label'].str.contains('Probing-IP sweep')]
print("Total numbers of Probing-Ipsweep : {}".format(len(train_Probing_Ipsweep)))

train_Probing_Nmap = train_data.loc[train_data['label'].str.contains('Probing-Nmap')]
print("Total numbers of Probing-Nmap : {}".format(len(train_Probing_Nmap)))

train_Probing_Port_sweep = train_data.loc[train_data['label'].str.contains('Probing-Port sweep')]
print("Total numbers of Probing-Port sweep : {}".format(len(train_Probing_Port_sweep)))


# In[6]:


num_train_Normal = [] 
num_train_DOS    = [] 
num_train_ProbIP = [] 
num_train_ProbNMAP = [] 
num_train_ProbPort = [] 

for i in range(len(train_Normal)):
    num_train_Normal.append(i)

for i in range(len(train_DOS_smurf)):
    num_train_DOS.append(i)
    
for i in range(len(train_Probing_Ipsweep)):
    num_train_ProbIP.append(i)
    
for i in range(len(train_Probing_Nmap)):
    num_train_ProbNMAP.append(i)
    
for i in range(len(train_Probing_Port_sweep)):
    num_train_ProbPort.append(i)
    


# In[7]:


random.shuffle(num_train_Normal)
random.shuffle(num_train_DOS)
random.shuffle(num_train_ProbIP)
random.shuffle(num_train_ProbNMAP)
random.shuffle(num_train_ProbPort)


# In[8]:


#重新給定index
train_Normal.index = range(len(train_Normal))
train_DOS_smurf.index = range(len(train_DOS_smurf))
train_Probing_Ipsweep.index = range(len(train_Probing_Ipsweep))
train_Probing_Nmap.index = range(len(train_Probing_Nmap))
train_Probing_Port_sweep.index = range(len(train_Probing_Port_sweep))


# In[9]:


#分成五組dataset 
train_Normal01 = train_Normal.iloc[0: 1784095, :]
train_Normal02 = train_Normal.iloc[1784095: 3568191, :]
train_Normal03 = train_Normal.iloc[3568191: 5352287, :]
train_Normal04 = train_Normal.iloc[5352287: 7136383, :]
train_Normal05 = train_Normal.iloc[7136383: 8920477, :]
print("total = {}".format(len(train_Normal01)+len(train_Normal02)+len(train_Normal03)+len(train_Normal04)+len(train_Normal05)))

train_DOS_smurf01 = train_DOS_smurf.iloc[0: 468, :]
train_DOS_smurf02 = train_DOS_smurf.iloc[468: 936, :]
train_DOS_smurf03 = train_DOS_smurf.iloc[936: 1404, :]
train_DOS_smurf04 = train_DOS_smurf.iloc[1404: 1873, :]
train_DOS_smurf05 = train_DOS_smurf.iloc[1873: 2345, :]
print("total = {}".format(len(train_DOS_smurf01)+len(train_DOS_smurf02)+len(train_DOS_smurf03)+len(train_DOS_smurf04)+len(train_DOS_smurf05)))

train_Probing_Ipsweep01 = train_Probing_Ipsweep.iloc[0: 48104, :]
train_Probing_Ipsweep02 = train_Probing_Ipsweep.iloc[48104: 96208, :]
train_Probing_Ipsweep03 = train_Probing_Ipsweep.iloc[96208: 144313, :]
train_Probing_Ipsweep04 = train_Probing_Ipsweep.iloc[144313: 192417, :]
train_Probing_Ipsweep05 = train_Probing_Ipsweep.iloc[192417: 240524, :]
print("total = {}".format(len(train_Probing_Ipsweep01)+len(train_Probing_Ipsweep02)+len(train_Probing_Ipsweep03)+len(train_Probing_Ipsweep04)+len(train_Probing_Ipsweep05)))

train_Probing_Nmap01 = train_Probing_Nmap.iloc[0: 165, :]
train_Probing_Nmap02 = train_Probing_Nmap.iloc[165: 330, :]
train_Probing_Nmap03 = train_Probing_Nmap.iloc[330: 495, :]
train_Probing_Nmap04 = train_Probing_Nmap.iloc[495: 660, :]
train_Probing_Nmap05 = train_Probing_Nmap.iloc[660: 829, :]
print("total = {}".format(len(train_Probing_Nmap01)+len(train_Probing_Nmap02)+len(train_Probing_Nmap03)+len(train_Probing_Nmap04)+len(train_Probing_Nmap05)))

train_Probing_Port_sweep01 = train_Probing_Port_sweep.iloc[0: 15457, :]
train_Probing_Port_sweep02 = train_Probing_Port_sweep.iloc[15457: 30914, :]
train_Probing_Port_sweep03 = train_Probing_Port_sweep.iloc[30914: 46371, :]
train_Probing_Port_sweep04 = train_Probing_Port_sweep.iloc[46371: 61829, :]
train_Probing_Port_sweep05 = train_Probing_Port_sweep.iloc[61829: 77289, :]
print("total = {}".format(len(train_Probing_Port_sweep01)+len(train_Probing_Port_sweep02)+len(train_Probing_Port_sweep03)+len(train_Probing_Port_sweep04)+len(train_Probing_Port_sweep05)))


# In[10]:


train_data_model01 = pd.concat([train_Normal01, train_DOS_smurf01, train_Probing_Ipsweep01, train_Probing_Nmap01, train_Probing_Port_sweep01], axis=0)
train_data_model02 = pd.concat([train_Normal01, train_DOS_smurf02, train_Probing_Ipsweep02, train_Probing_Nmap02, train_Probing_Port_sweep02], axis=0)
train_data_model03 = pd.concat([train_Normal01, train_DOS_smurf03, train_Probing_Ipsweep03, train_Probing_Nmap03, train_Probing_Port_sweep03], axis=0)
train_data_model04 = pd.concat([train_Normal01, train_DOS_smurf04, train_Probing_Ipsweep04, train_Probing_Nmap04, train_Probing_Port_sweep04], axis=0)
train_data_model05 = pd.concat([train_Normal01, train_DOS_smurf05, train_Probing_Ipsweep05, train_Probing_Nmap05, train_Probing_Port_sweep05], axis=0)


# In[11]:


#store to csv files 
train_data_model01.to_csv('/home/shaowen0213/NAD/training set/new_construction/train_data_model01.csv', index=False, encoding='utf-8-sig')
train_data_model02.to_csv('/home/shaowen0213/NAD/training set/new_construction/train_data_model02.csv', index=False, encoding='utf-8-sig')
train_data_model03.to_csv('/home/shaowen0213/NAD/training set/new_construction/train_data_model03.csv', index=False, encoding='utf-8-sig')
train_data_model04.to_csv('/home/shaowen0213/NAD/training set/new_construction/train_data_model04.csv', index=False, encoding='utf-8-sig')
train_data_model05.to_csv('/home/shaowen0213/NAD/training set/new_construction/train_data_model05.csv', index=False, encoding='utf-8-sig')


# In[ ]:




