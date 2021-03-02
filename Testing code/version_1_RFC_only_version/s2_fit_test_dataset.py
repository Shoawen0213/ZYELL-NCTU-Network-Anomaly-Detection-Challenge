#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import sys
import time
import numpy as np
import pandas as pd
import socket
import struct
from tqdm.notebook import tqdm
from time import sleep
import datacompy

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


# In[11]:


def fit_test_data(dataframe):
    # use dummy on "app"
    dataframe_dum_app = dataframe.join(pd.get_dummies(dataframe.app))
    print(dataframe_dum_app) 
    
    #delete ['time', 'src', 'dst', 'app']
    dataframe_dum_app = dataframe_dum_app.drop(columns = ['time', 'src', 'dst', 'app']) 
    print(dataframe_dum_app.columns)
    
    # orginal dataframe (66 features)
    org_df = pd.DataFrame(columns = ['int_time', 'int_src', 'int_dst', 'spt', 'dpt', 'duration',
       'out (bytes)', 'in (bytes)', 'proto', 'cnt_dst', 'cnt_src',
       'cnt_serv_src', 'cnt_serv_dst', 'cnt_dst_slow', 'cnt_src_slow',
       'cnt_serv_src_slow', 'cnt_serv_dst_slow', 'cnt_dst_conn',
       'cnt_src_conn', 'cnt_serv_src_conn', 'cnt_serv_dst_conn',
       'aim', 'auth', 'bgp', 'bootpc', 'bootps', 'domain', 'finger', 'ftp',
       'h323', 'http', 'https', 'icmp', 'icq', 'igmp', 'irc', 'isakmp',
       'microsoft-ds', 'msn', 'netbios-dgm', 'netbios-ns', 'netbios-ssn',
       'news', 'nfs', 'nntp', 'others', 'pop3', 'pptp', 'rcmd', 'real-audio',
       'rexec', 'rlogin', 'roadrunner', 'rtsp', 'sftp', 'smtp', 'snmp',
       'snmp-trap', 'sql-net', 'ssdp', 'ssh', 'syslog', 'tacacs', 'telnet',
       'tftp', 'vdolive'])
    org_df # 0 rows × 66 columns
    
    compare = datacompy.Compare(dataframe_dum_app, org_df, on_index = True)
    print("compare Result:", compare.report())
    print("缺少的為 :{}".format(compare.df2_unq_columns()))
    lost_list = compare.df2_unq_columns()
    
    lost_zero = np.zeros([len(dataframe_dum_app)    , len(lost_list)])
    concact_lost_zero_df = pd.DataFrame(lost_zero, columns = lost_list) 
    
    dataframe_66col =  pd.concat([dataframe_dum_app, concact_lost_zero_df], axis=1)
    
    
    print("-------------------")
    print("start to predict...")
    print("-------------------")
    rfc_pred = rfc.predict(dataframe_66col)
    print("predict result : ", rfc_pred)
    print("--------------------------------")
    print("shape of data : ", rfc_pred.shape)
    unique, counts = np.unique(rfc_pred, return_counts=True)
    print("data contains:{} ".format(dict(zip(unique, counts))))
    
    # predict result transform into df
    rfc_pred_df = {"label": rfc_pred}
    rfc_pred_df = pd.DataFrame(rfc_pred_df)
    rfc_pred_df  
    
     for i in tqdm(range(len(rfc_pred_df))):
#    for i in tqdm(range(100)):
        if rfc_pred_df.label[i] == 0:
            rfc_pred_df.label[i] = 'Normal'
        elif rfc_pred_df.label[i] == 1:
            rfc_pred_df.label[i] = 'Probing-Nmap'
        elif rfc_pred_df.label[i] == 2:
            rfc_pred_df.label[i] = 'Probing-Port sweep'
        elif rfc_pred_df.label[i] == 3:
            rfc_pred_df.label[i] = 'Probing-IP sweep'
        else :
            rfc_pred_df.label[i] = 'DDOS-smurf'
        sleep(0.01)
    
    print(rfc_pred_df.shape)
    
    return rfc_pred_df


# In[5]:


print("now move to ...")
now_path = os.getcwd()
print(now_path)


# In[6]:


#load model
rfc = joblib.load('./RFC_only.pkl')


# In[7]:

print("//------------------------// ")
print("//  Now for 0123_firewall // ")
print("//------------------------// ")


# prediction for test_0123
file_path = './transformed/0123_transformed_solo.csv'
test_data_0123 = pd.read_csv(file_path)
test_data_0123  #3601186  rows × 25 columns


# In[10]:


rfc_pred_df = fit_test_data(test_data_0123)

test_data_0123_orgin = test_data_0123.drop(columns = ['int_time', 'int_src', 'int_dst'])

test_0123_concat =  pd.concat([test_data_0123_orgin, rfc_pred_df], axis=1)
test_0123_concat #3601186 rows × 23 columns

#store to csv files 
test_0123_concat.to_csv('./131_OASIS LAB_0123_firewall.csv', index=False, encoding='utf-8-sig')


# In[15]:


print("//------------------------// ")
print("//  Now for 0124_firewall // ")
print("//------------------------// ")

# prediction for test_0124
file_path = './transformed/0124_transformed_solo.csv'
test_data_0124= pd.read_csv(file_path)
test_data_0124 #2050710 rows × 25 columns


# In[16]:


rfc_pred_df = fit_test_data(test_data_0124)

test_data_0124_orgin = test_data_0124.drop(columns = ['int_time', 'int_src', 'int_dst'])

test_0124_concat =  pd.concat([test_data_0124_orgin, rfc_pred_df], axis=1)
test_0124_concat #2050710 rows × 23 columns

#store to csv files 
test_0124_concat.to_csv('./131_OASIS LAB_0124_firewall.csv', index=False, encoding='utf-8-sig')


# In[17]:


print("//------------------------// " )
print("//  Now for 0125_firewall // " )
print("//------------------------// " )


# prediction for test_0125
file_path = './transformed/0125_transformed_solo.csv'
test_data_0125 = pd.read_csv(file_path)
test_data_0125 #2120819 rows × 25 columns


# In[18]:


rfc_pred_df = fit_test_data(test_data_0125)

test_data_0125_orgin = test_data_0125.drop(columns = ['int_time', 'int_src', 'int_dst'])

test_0125_concat =  pd.concat([test_data_0125_orgin, rfc_pred_df], axis=1)
print(test_0125_concat.shape) #2120819 rows × 23 columns

#store to csv files 
test_0125_concat.to_csv('./131_OASIS LAB_0125_firewall.csv', index=False, encoding='utf-8-sig')


# In[20]:


print("//------------------------// " )
print("//  Now for 0126_firewall // " )
print("//------------------------// " )

# prediction for test_0126
file_path = './transformed/0126_transformed_solo.csv'
test_data_0126 = pd.read_csv(file_path)
test_data_0126 #5517815 rows × 25 columns


# In[22]:


rfc_pred_df = fit_test_data(test_data_0126)

test_data_0126_orgin = test_data_0126.drop(columns = ['int_time', 'int_src', 'int_dst'])

test_0126_concat =  pd.concat([test_data_0126_orgin, rfc_pred_df], axis=1)
print(test_0126_concat.shape) #2120819 rows × 23 columns

#store to csv files 
test_0126_concat.to_csv('./131_OASIS LAB_0126_firewall.csv', index=False, encoding='utf-8-sig')

