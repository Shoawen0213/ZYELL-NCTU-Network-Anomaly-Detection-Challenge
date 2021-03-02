#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import glob
import sys
import time
import numpy as np
import pandas as pd
import socket
import struct
import datacompy
from tqdm.notebook import tqdm
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder


# In[18]:


print("now move to ...")
now_path = os.getcwd()
print(now_path)


# In[19]:


print("Loading model from : {}".format(os.getcwd()))
#load model
xgbc = joblib.load('./XGB_only.pkl')


# In[20]:


print("fetach original features...")

# orginal dataframe (66 features)
org_list = ['int_time', 'int_src', 'int_dst', 'spt', 'dpt', 'duration',
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
       'tftp', 'vdolive']
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
print(org_df.columns) # 0 rows × 66 columns
print("//----------------------------------------------//")


# In[21]:


def fit_data(dataframe, org_list,  org_df):
    # use dummy on "app"
    dataframe_orgi = dataframe.drop(columns = ['int_time', 'int_src', 'int_dst'])
    dataframe_dum_app = dataframe.join(pd.get_dummies(dataframe.app))
    print("shape of dataframe after dummy:{}".format(dataframe_dum_app.shape)) 
    
    #delete ['time', 'src', 'dst', 'app']
    dataframe_dum_app = dataframe_dum_app.drop(columns = ['time', 'src', 'dst', 'app'])

    print("Start to check lost app feature...")
    compare = datacompy.Compare(dataframe_dum_app, org_df, on_index = True)
    print("compare Result:", compare.report())
    print("//----------------------------------------------//")
    print("缺少的為 :{}".format(compare.df2_unq_columns()))
    lost_list = compare.df2_unq_columns()
    
    lost_zero = np.zeros([len(dataframe_dum_app) , len(lost_list)]) 
    concact_lost_zero_df = pd.DataFrame(lost_zero, columns = lost_list)
    print("shape of concact_lost_zero_df:{}".format(concact_lost_zero_df.shape)) 
    print("//----------------------------------------------//")
    dataframe_66col =  pd.concat([dataframe_dum_app, concact_lost_zero_df], axis=1)
    print("shape of dataframe_66col:{}".format(dataframe_66col.shape))
    print("//----------------------------------------------//")
    
    print("Check if there's any missing...")
    dataframe_66col_reload = dataframe_66col[org_list]
    compare = datacompy.Compare(dataframe_66col, dataframe_66col_reload, on_index = True)
    print("compare Result:", compare.report())
    print("缺少的為 :{}".format(compare.df2_unq_columns()))
    if compare.df2_unq_columns() == str(set()):
        print("There;s just completely the same.")
    else :
        print("{} is missing".format(compare.df2_unq_columns()))
    print("//----------------------------------------------//")
    
    print("Start to PREDICT!...")
    xgbc_pred = xgbc.predict(dataframe_66col_reload)
    print("predict result : ", xgbc_pred)
    print("--------------------------------")
    print("shape of xgbc_pred data : ", xgbc_pred.shape)
    unique, counts = np.unique(xgbc_pred, return_counts=True)
    print("xgbc_pred data contains:{} ".format(dict(zip(unique, counts))))
    
    label_list = ['Normal', 'DOS-smurf', 'Probing-Ipsweep', 'Probing-Nmap', 'Probing-Port sweep']
    print("label_list : {}".format(label_list))
    #label encode
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(label_list)
    print("label_list after label encode : {}".format(y))
    print("//--------------------------------------------//")
    
    #label encode revise
    transform = labelencoder.inverse_transform(xgbc_pred)
    df = pd.DataFrame(transform, columns = ['label'])
    print("Label revised : {}".format(df))
    print("shape of revised label : ", df.shape)
    unique, counts = np.unique(df, return_counts=True)
    print("xgbc_predict contains:{}".format(dict(zip(unique, counts))))
    
    print("//--------------------------------------------//")
    print("shape of orginal data : ", dataframe_orgi.shape)
    
    print("//--------------------------------------------//")
    data_concat =  pd.concat([dataframe_orgi, df], axis=1)
    print("shape of data_concat : ", data_concat.shape)
    
    return data_concat


# In[22]:


print("//------------------------// ")
print("//  Now for 0123_firewall // ")
print("//------------------------// ")

# prediction for test_0123
file_path = './transformed/0123_transformed_solo.csv'
test_data_0123 = pd.read_csv(file_path)
test_data_0123  #3601186  rows × 25 columns


# In[ ]:


concat_data = fit_data(test_data_0123, org_list, org_df)
print("Shape of data to submit : {}".format(concat_data.shape))
print(concat_data.columns)

#store to csv files 
concat_data.to_csv('./131_OASIS LAB_02_0123_firewall.csv', index=False, encoding='utf-8-sig')

print("Finish Storing!")


# In[ ]:


print("//------------------------// ")
print("//  Now for 0124_firewall // ")
print("//------------------------// ")

# prediction for test_0124
file_path = './transformed/0124_transformed_solo.csv'
test_data_0124 = pd.read_csv(file_path)
test_data_0124  #3601186  rows × 25 columns


# In[ ]:


concat_data = fit_data(test_data_0124, org_list, org_df)
print("Shape of data to submit : {}".format(concat_data.shape))
print(concat_data.columns)

#store to csv files 
concat_data.to_csv('./131_OASIS LAB_02_0124_firewall.csv', index=False, encoding='utf-8-sig')

print("Finish Storing!")


# In[ ]:


print("//------------------------// ")
print("//  Now for 0125_firewall // ")
print("//------------------------// ")

# prediction for test_0125
file_path = './transformed/0125_transformed_solo.csv'
test_data_0125 = pd.read_csv(file_path)
test_data_0125  #3601186  rows × 25 columns


# In[ ]:


concat_data = fit_data(test_data_0125, org_list, org_df)
print("Shape of data to submit : {}".format(concat_data.shape))
print(concat_data.columns)

#store to csv files 
concat_data.to_csv('./131_OASIS LAB_02_0125_firewall.csv', index=False, encoding='utf-8-sig')

print("Finish Storing!")


# In[ ]:


print("//------------------------// ")
print("//  Now for 0126_firewall // ")
print("//------------------------// ")

# prediction for test_0126
file_path = './transformed/0126_transformed_solo.csv'
test_data_0126 = pd.read_csv(file_path)
test_data_0126  #3601186  rows × 25 columns


# In[ ]:


concat_data = fit_data(test_data_0126, org_list, org_df)
print("Shape of data to submit : {}".format(concat_data.shape))
print(concat_data.columns)

#store to csv files 
concat_data.to_csv('./131_OASIS LAB_02_0126_firewall.csv', index=False, encoding='utf-8-sig')

print("Finish Storing!")

