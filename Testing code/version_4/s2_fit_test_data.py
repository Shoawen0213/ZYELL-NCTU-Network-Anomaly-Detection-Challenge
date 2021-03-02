#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import glob
import sys
import time
import numpy as np
import pandas as pd
import datacompy
import csv
from time import sleep
from random import uniform
from tqdm import tqdm
from scipy import stats 
from pandas import  DataFrame
from tqdm.notebook import tqdm

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder


# In[4]:


print("now in...")
now_path = os.getcwd()
print(now_path)


# In[5]:


print("Loading model...")
m=0
#load model
model1 = joblib.load('./XGB_dummy_app.pkl')
m+=1
model2 = joblib.load('./rfc_dum_app.pkl')
m+=1
model3 = joblib.load('./XGB_dummy_app.pkl')
m+=1
model4 = joblib.load('./rfc_dum_app.pkl')
m+=1
model5 = joblib.load('./XGB_dummy_app.pkl')
m+=1
model6 = joblib.load('./rfc_dum_app.pkl')
m+=1
model7 = joblib.load('./XGB_dummy_app.pkl')
m+=1
print("Finisg loading...")
print("We got {} sub-model".format(m))


# In[6]:


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
org_df # 0 rows × 66 columns


# In[7]:


def main(dataframe):
    print('//---------------------//')
    print("// now using model1... //")
    print('//---------------------//')
    model1_ = pd.DataFrame(fit_data_dum(dataframe, org_list,  org_df, model1))
    print('//---------------------//')
    print("// now using model2... //")
    print('//---------------------//')
    model2_ = pd.DataFrame(fit_data_dum(dataframe, org_list,  org_df, model2))
    print('//---------------------//')
    print("// now using model3... //")
    print('//---------------------//')
    model3_ = pd.DataFrame(fit_data_dum(dataframe, org_list,  org_df, model3))
    print('//---------------------//')
    print("// now using model4... //")
    print('//---------------------//')
    model4_ = pd.DataFrame(fit_data_dum(dataframe, org_list,  org_df, model4))
    print('//---------------------//')
    print("// now using model5... //")
    print('//---------------------//')
    model5_ = pd.DataFrame(fit_data_dum(dataframe, org_list,  org_df, model5))
    print('//---------------------//')
    print("// now using model6... //")
    print('//---------------------//')
    model6_ = pd.DataFrame(fit_data_dum(dataframe, org_list,  org_df, model6))
    print('//---------------------//')
    print("// now using model7... //")
    print('//---------------------//')
    model7_ = pd.DataFrame(fit_data_dum(dataframe, org_list,  org_df, model7))
    
    print('//-------------------------//')
    print("// concat all predict data //")
    print('//-------------------------//')
    concat_all_label = pd.concat([model1_, model2_, model3_, model4_, model5_, model6_, model7_], axis=1)

    return concat_all_label


def fit_data_no_dum(dataframe, model):
    dataframe_drop = dataframe[["int_time", 'int_src', 'int_dst', 'spt', 'dpt', 'duration', 'out (bytes)', 'in (bytes)', 'proto', 'cnt_dst', 'cnt_src', 'cnt_serv_src', 'cnt_serv_dst', 'cnt_dst_slow', 'cnt_src_slow', 'cnt_serv_src_slow', 'cnt_serv_dst_slow', 'cnt_dst_conn', 'cnt_src_conn', 'cnt_serv_src_conn', 'cnt_serv_dst_conn']]
    dataframe_orgi = dataframe.drop(columns = ['int_time', 'int_src', 'int_dst'])
    
    model_predict = model.predict(dataframe_drop)
    print("shape of xgbc_predict : ", model_predict.shape)
    unique, counts = np.unique(model_predict, return_counts=True)
    print("xgbc_predict contains:{}".format(dict(zip(unique, counts))))
    
    return model_predict

def fit_data_dum(dataframe, org_list,  org_df, model):
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
    model_pred = model.predict(dataframe_66col_reload)
    print("predict result : ", model_pred)
    print("--------------------------------")
    print("shape of xgbc_pred data : ", model_pred.shape)
    unique, counts = np.unique(model_pred, return_counts=True)
    print("pred data contains:{} ".format(dict(zip(unique, counts))))
    
    return model_pred

def dum_inverse(df):
    label_list = ['Normal', 'DDOS-smurf', 'Probing-IP sweep', 'Probing-Nmap', 'Probing-Port sweep']
    print("label_list : {}".format(label_list))
    #label encode
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(label_list)
    print("label_list after label encode : {}".format(y))
    print("//--------------------------------------------//")
    
    #label encode revise
    transform = labelencoder.inverse_transform(df)
    df_inverse = pd.DataFrame(transform, columns = ['label'])
    print("Label revised : {}".format(df_inverse))
    print("shape of revised label : ", df_inverse.shape)
    unique, counts = np.unique(df_inverse, return_counts=True)
    print("xgbc_predict contains:{}".format(dict(zip(unique, counts))))
    
    return df_inverse

def concat(dataframe, concat_df):
    dataframe_orgi = dataframe.drop(columns = ['int_time', 'int_src', 'int_dst'])
    print("//--------------------------------------------//")
    print("shape of orginal data : ", dataframe_orgi.shape)
    
    print("//--------------------------------------------//")
    data_concat =  pd.concat([dataframe_orgi, concat_df], axis=1)
    print("shape of data_concat : ", data_concat.shape)
   
    return data_concat


# In[8]:


# prediction for test_0123
print('//----------------------------//')
print('// NOW Fitting test data 0123 //')
print('//----------------------------//')
file_path = './transformed/0123_transformed_solo.csv'
test_data_0123 = pd.read_csv(file_path)
test_data_0123  #3601186  rows × 25 columns


# In[9]:


voted_label = main(test_data_0123)

#store to csv files 
voted_label.to_csv('./0123_concat_predict.csv', index=False, encoding='utf-8-sig')
print("Finish store tmp csv...")

print("--------------------------------")
print("        Start Voting!!          ")
print("--------------------------------")
progress = tqdm(total=3601186   , ncols = 80)
with open("./0123_concat_predict.csv", newline='', encoding="utf-8") as csvfile:
    df = []
    reader = csv.reader(csvfile)
    for line in reader:
        #sleep(0.01)
        #print (line[:])
        mo = stats.mode(line)[0][0]
        #print(mo)
        df.append(mo)
        del line
        del mo
        progress.update(1)
print(df)
progress.close()

df = list(map(int, df))
print("length of df : ", len(df))
# print(df)
print("--------------------------------")
voted_label_df = pd.DataFrame(df, columns=['label'])
print("--------------------------------")
voted_label_df = voted_label_df[0:len(voted_label_df)-1]
print("shape of voted_label_df" , voted_label_df.shape)
print("--------------------------------")

unique, counts = np.unique(voted_label_df, return_counts=True)
print("pred data contains:{} ".format(dict(zip(unique, counts))))
print("--------------------------------")

label_inverse = dum_inverse(voted_label_df)
label_inverse

concat_data = concat(test_data_0123, label_inverse)
print("shape of concat_data" , concat_data.shape)

#store to csv files 
concat_data.to_csv('./131_OASIS LAB_04_0123_firewall.csv', index=False, encoding='utf-8-sig')

print(" 0123 Finish!")


# In[12]:


# prediction for test_0124
print('//----------------------------//')
print('// NOW Fitting test data 0124 //')
print('//----------------------------//')
file_path = './transformed/0124_transformed_solo.csv'
test_data_0124 = pd.read_csv(file_path)
test_data_0124  #2050710  rows × 25 columns


# In[ ]:


voted_label = main(test_data_0124)

#store to csv files 
voted_label.to_csv('./0124_concat_predict.csv', index=False, encoding='utf-8-sig')
print("Finish store tmp csv...")

print("--------------------------------")
print("        Start Voting!!          ")
print("--------------------------------")
progress = tqdm(total=2050710 , ncols = 80)
with open("./0124_concat_predict.csv", newline='', encoding="utf-8") as csvfile:
    df = []
    reader = csv.reader(csvfile)
    #for line in tqdm(reader, unit="keystrokes", desc="Loading", position=1, disable=False):
    for line in reader:
        #sleep(0.01)
        #print (line[:])
        mo = stats.mode(line)[0][0]
        #print(mo)
        df.append(mo)
        del line
        del mo
        progress.update(1)
print(df)
progress.close()

df = list(map(int, df))
print("length of df : ", len(df))
# print(df)
print("--------------------------------")
voted_label_df = pd.DataFrame(df, columns=['label'])
print("--------------------------------")
voted_label_df = voted_label_df[0:len(voted_label_df)-1]
print("shape of voted_label_df" , voted_label_df.shape)
print("--------------------------------")

unique, counts = np.unique(voted_label_df, return_counts=True)
print("pred data contains:{} ".format(dict(zip(unique, counts))))
print("--------------------------------")

label_inverse = dum_inverse(voted_label_df)
label_inverse

concat_data = concat(test_data_0124, label_inverse)
print("shape of concat_data" , concat_data.shape)

#store to csv files 
concat_data.to_csv('./131_OASIS LAB_04_0124_firewall.csv', index=False, encoding='utf-8-sig')

print(" 0124 Finish!")


# In[11]:


# prediction for test_0125
print('//----------------------------//')
print('// NOW Fitting test data 0125 //')
print('//----------------------------//')
file_path = './transformed/0125_transformed_solo.csv'
test_data_0125 = pd.read_csv(file_path)
test_data_0125  #2120819  rows × 25 columns


# In[ ]:


voted_label = main(test_data_0125)

#store to csv files 
voted_label.to_csv('./0125_concat_predict.csv', index=False, encoding='utf-8-sig')
print("Finish store tmp csv...")

print("--------------------------------")
print("        Start Voting!!          ")
print("--------------------------------")
progress = tqdm(total=2120819 , ncols = 80)
with open("./0125_concat_predict.csv", newline='', encoding="utf-8") as csvfile:
    df = []
    reader = csv.reader(csvfile)
    for line in reader:
        mo = stats.mode(line)[0][0]
        df.append(mo)
        del line
        del mo
        progress.update(1)
print(df)
progress.close()

df = list(map(int, df))
print("length of df : ", len(df))
# print(df)
print("--------------------------------")
voted_label_df = pd.DataFrame(df, columns=['label'])
print("--------------------------------")
voted_label_df = voted_label_df[0:len(voted_label_df)-1]
print("shape of voted_label_df" , voted_label_df.shape)
print("--------------------------------")

unique, counts = np.unique(voted_label_df, return_counts=True)
print("pred data contains:{} ".format(dict(zip(unique, counts))))
print("--------------------------------")

label_inverse = dum_inverse(voted_label_df)
label_inverse

concat_data = concat(test_data_0125, label_inverse)
print("shape of concat_data" , concat_data.shape)

#store to csv files 
concat_data.to_csv('./131_OASIS LAB_04_0125_firewall.csv', index=False, encoding='utf-8-sig')

print(" 0125 Finish!")


# In[13]:


# prediction for test_0126
print('//----------------------------//')
print('// NOW Fitting test data 0126 //')
print('//----------------------------//')
file_path = './transformed/0126_transformed_solo.csv'
test_data_0126 = pd.read_csv(file_path)
test_data_0126  #5517815  rows × 25 columns


# In[ ]:


voted_label = main(test_data_0126)

#store to csv files 
voted_label.to_csv('./0126_concat_predict.csv', index=False, encoding='utf-8-sig')
print("Finish store tmp csv...")

print("--------------------------------")
print("        Start Voting!!          ")
print("--------------------------------")
progress = tqdm(total=5517815  , ncols = 80)
with open("./0126_concat_predict.csv", newline='', encoding="utf-8") as csvfile:
    df = []
    reader = csv.reader(csvfile)
    for line in reader:
        #sleep(0.01)
        #print (line[:])
        mo = stats.mode(line)[0][0]
        #print(mo)
        df.append(mo)
        del line
        del mo
        progress.update(1)
print(df)
progress.close()

df = list(map(int, df))
print("length of df : ", len(df))
# print(df)
print("--------------------------------")
voted_label_df = pd.DataFrame(df, columns=['label'])
print("--------------------------------")
voted_label_df = voted_label_df[0:len(voted_label_df)-1]
print("shape of voted_label_df" , voted_label_df.shape)
print("--------------------------------")

unique, counts = np.unique(voted_label_df, return_counts=True)
print("pred data contains:{} ".format(dict(zip(unique, counts))))
print("--------------------------------")

label_inverse = dum_inverse(voted_label_df)
label_inverse

concat_data = concat(test_data_0126, label_inverse)
print("shape of concat_data" , concat_data.shape)

#store to csv files 
concat_data.to_csv('./131_OASIS LAB_04_0126_firewall.csv', index=False, encoding='utf-8-sig')

print(" 0126 Finish!")
