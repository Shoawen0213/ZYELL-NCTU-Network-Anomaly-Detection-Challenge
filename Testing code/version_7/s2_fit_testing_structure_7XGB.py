#!/usr/bin/env python
# coding: utf-8

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

print("now in...")
now_path = os.getcwd()
print(now_path)

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

print("Loading model...")
m=0
n=0
#load model
model1 = joblib.load('./rfc_0306_slice_all_ver0.pkl')
m+=1
model2 = joblib.load('./rfc_0306_slice_all_ver2.pkl')
m+=1
model3 = joblib.load('./rfc_0306_slice_all_ver3.pkl')
m+=1
model4 = joblib.load('./rfc_0306_slice_all_ver4.pkl')
m+=1
model5 = joblib.load('./xgb_0306_slice_all_ver4.pkl')
n+=1
model6 = joblib.load('./xgb_0306_slice_all_ver5.pkl')
n+=1
model7 = joblib.load('./xgb_0306_slice_all_ver6.pkl')
n+=1
model8 = joblib.load('./xgb_0306_slice_all_ver0.pkl')
n+=1
model9 = joblib.load('./xgb_0306_slice_all_ver1.pkl')
n+=1
model10 = joblib.load('./xgb_0306_slice_all_ver2.pkl')
n+=1
model11 = joblib.load('./xgb_0306_slice_all_ver3.pkl')
n+=1

print("Finisg loading...")
print("We got {} sub-model".format(m+n))
print("contain {} 's RFC and {}'s XGB.".format(m, n))

def main(dataframe):
    print('//---------------------//')
    print("// now using model1... //")
    print('//---------------------//')
    model1_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model1)
    model1_pred_prob_df = pd.DataFrame(model1_pred_prob)
    del model1_pred_prob
    
    print('//---------------------//')
    print("// now using model2... //")
    print('//---------------------//')
    model2_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model2)
    model2_pred_prob_df = pd.DataFrame(model2_pred_prob)
    del model2_pred_prob
   
    print('//---------------------//')
    print("// now using model3... //")
    print('//---------------------//')
    model3_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model3)
    model3_pred_prob_df = pd.DataFrame(model3_pred_prob)
    del model3_pred_prob

    print('//---------------------//')
    print("// now using model4... //")
    print('//---------------------//')
    model4_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model4)
    model4_pred_prob_df = pd.DataFrame(model4_pred_prob)
    del model4_pred_prob
     
    print('//---------------------//')
    print("// now using model5... //")
    print('//---------------------//')
    model5_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model5)
    model5_pred_prob_df = pd.DataFrame(model5_pred_prob)
    del model5_pred_prob
     
     
    print('//---------------------//')
    print("// now using model6... //")
    print('//---------------------//')
    model6_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model6)
    model6_pred_prob_df = pd.DataFrame(model6_pred_prob)
    del model6_pred_prob
     
    
    print('//---------------------//')
    print("// now using model7... //")
    print('//---------------------//')
    model7_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model7)
    model7_pred_prob_df = pd.DataFrame(model7_pred_prob)
    del model7_pred_prob
    
    print('//---------------------//')
    print("// now using model8... //")
    print('//---------------------//')
    model8_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model8)
    model8_pred_prob_df = pd.DataFrame(model8_pred_prob)
    del model8_pred_prob
    
    print('//---------------------//')
    print("// now using model9... //")
    print('//---------------------//')
    model9_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model9)
    model9_pred_prob_df = pd.DataFrame(model9_pred_prob)
    del model9_pred_prob
    
    print('//---------------------//')
    print("// now using model10... //")
    print('//---------------------//')
    model10_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model10)
    model10_pred_prob_df = pd.DataFrame(model10_pred_prob)
    del model10_pred_prob
    
    print('//---------------------//')
    print("// now using model11... //")
    print('//---------------------//')
    model11_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model11)
    model11_pred_prob_df = pd.DataFrame(model11_pred_prob)
    del model11_pred_prob

    
    print('//-------------------------//')
    print("// concat all predict data //")
    print('//-------------------------//')  
    concat_pred_prob_df = pd.concat([model1_pred_prob_df, model2_pred_prob_df, model3_pred_prob_df, model4_pred_prob_df, model5_pred_prob_df,
                                    model6_pred_prob_df, model7_pred_prob_df, model8_pred_prob_df, model9_pred_prob_df, model10_pred_prob_df, model11_pred_prob_df], axis=1)
    
    del model1_pred_prob_df, model2_pred_prob_df, model3_pred_prob_df, model4_pred_prob_df, model5_pred_prob_df
    del model6_pred_prob_df, model7_pred_prob_df, model8_pred_prob_df, model9_pred_prob_df, model10_pred_prob_df
    del model11_pred_prob_df
    
    return concat_pred_prob_df


def fit_data_dum(dataframe, org_list, org_df, model):
    # use dummy on "app"
    dataframe_orgi = dataframe.drop(columns = ['int_time', 'int_src', 'int_dst'])
    dataframe_dum_app = dataframe.join(pd.get_dummies(dataframe.app))
    print("shape of dataframe after dummy:{}".format(dataframe_dum_app.shape)) 
    
    #delete ['time', 'src', 'dst', 'app']
    dataframe_dum_app = dataframe_dum_app.drop(columns = ['time', 'src', 'dst', 'app'])

    print("Start to check lost app feature...")
    compare = datacompy.Compare(dataframe_dum_app, org_df, on_index = True)
    print("compare Result --> lost :", compare.df2_unq_columns())
    print("//----------------------------------------------//")
    lost_list = compare.df2_unq_columns()
    
    lost_zero = np.zeros([len(dataframe_dum_app) , len(lost_list)]) 
    concact_lost_zero_df = pd.DataFrame(lost_zero, columns = lost_list)
    del lost_zero
    del lost_list
    
    print("shape of concact_lost_zero_df:{}".format(concact_lost_zero_df.shape)) 
    print("//----------------------------------------------//")
    dataframe_66col =  pd.concat([dataframe_dum_app, concact_lost_zero_df], axis=1)
    print("shape of dataframe_66col:{}".format(dataframe_66col.shape))
    print("//----------------------------------------------//")
    
    print("Check if there's any missing...")
    dataframe_66col_reload = dataframe_66col[org_list]
    compare = datacompy.Compare(dataframe_66col, dataframe_66col_reload, on_index = True)
    
    del dataframe_66col
    
    print("compare Result:", compare.report())
    print("//----------------------------------------------//")
    
    print("Start to PREDICT!...")
    model_pred_prob = model.predict_proba(dataframe_66col_reload)
    print("predict result : ", model_pred_prob)
    print("--------------------------------")
    print("shape of predict data : ", model_pred_prob.shape)
    
    return model_pred_prob
    
def dum_inverse(df):
    label_list = ['Normal', 'DDOS-smurf', 'Probing-IP sweep', 'Probing-Nmap', 'Probing-Port sweep']
    print("label_list : {}".format(label_list))
    #label encode
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(label_list)
    print("label_list after label encode : {}".format(y))
    print("//--------------------------------------------//")
    del y 
    
    #label encode revise
    transform = labelencoder.inverse_transform(df)
    df_inverse = pd.DataFrame(transform, columns = ['label'])
    print("Label revised : {}".format(df_inverse))
    print("shape of revised label : ", df_inverse.shape)
    unique, counts = np.unique(df_inverse, return_counts=True)
    print("xgbc_predict contains:{}".format(dict(zip(unique, counts))))
    
    del transform
    
    return df_inverse

def concat(dataframe, concat_df):
    dataframe_orgi = dataframe.drop(columns = ['int_time', 'int_src', 'int_dst'])
    print("//--------------------------------------------//")
    print("shape of orginal data : ", dataframe_orgi.shape)
    
    print("//--------------------------------------------//")
    data_concat =  pd.concat([dataframe_orgi, concat_df], axis=1)
    print("shape of data_concat : ", data_concat.shape)
    
    del dataframe_orgi
   
    return data_concat

print("//********************************//")
print("//    Loading stage 2 model...    //")
print("//********************************//")

model_s2 = joblib.load('./0306_final_stage_rf_clfver_7XGB_ver2.pkl')

print(" Loading model Successes! ")

# prediction for test_0123
print("//*******************************************//")
print("//*      Now predict for test data 0123     *//")
print("//*               Now using...              *//")
print("//*    7XGB + 4RFC + RandomForestClassifier *//")
print("//*******************************************//")
file_path = './transformed/0123_transformed_solo.csv'
test_data_0123 = pd.read_csv(file_path)

pred_prob_0123 = main(test_data_0123)

print("------------------------------------------------")

_predict = model_s2.predict(pred_prob_0123)
print("shape of _predict : ", _predict.shape)
unique, counts = np.unique(_predict, return_counts=True)
print("_predict contains:{}".format(dict(zip(unique, counts))))

print("------------------------------------------------")
print("//    Now doing label encode inverse    //")

_predict_inverse = dum_inverse(_predict)
print("shape of _predict_inverse : ", _predict_inverse.shape)
unique, counts = np.unique(_predict_inverse, return_counts=True)
print("_predict_inverse contains:{}".format(dict(zip(unique, counts))))

del _predict
print("------------------------------------------------")
concat_data = concat(test_data_0123, _predict_inverse)
print("shape of concat_data" , concat_data.shape)

print("//    Now Store the result    //")
#store to csv files 
concat_data.to_csv('./131_OASIS LAB_07_0123_firewall.csv', index=False, encoding='utf-8-sig')

del _predict_inverse
del concat_data
del test_data_0123

print(" 0123 Finish!")

# prediction for test_0124
print("//*******************************************//")
print("//*      Now predict for test data 0124     *//")
print("//*               Now using...              *//")
print("//*    7XGB + 4RFC + RandomForestClassifier *//")
print("//*******************************************//")
file_path = './transformed/0124_transformed_solo.csv'
test_data_0124 = pd.read_csv(file_path)

pred_prob_0124 = main(test_data_0124)

_predict = model_s2.predict(pred_prob_0124)
print("shape of _predict : ", _predict.shape)
unique, counts = np.unique(_predict, return_counts=True)
print("_predict contains:{}".format(dict(zip(unique, counts))))
print("------------------------------------------------")
print("//    Now doing label encode inverse    //")

_predict_inverse = dum_inverse(_predict)
print("shape of _predict_inverse : ", _predict_inverse.shape)
unique, counts = np.unique(_predict_inverse, return_counts=True)
print("_predict_inverse contains:{}".format(dict(zip(unique, counts))))

del _predict
print("------------------------------------------------")
concat_data = concat(test_data_0124, _predict_inverse)
print("shape of concat_data" , concat_data.shape)

print("//    Now Store the result    //")
#store to csv files 
concat_data.to_csv('./131_OASIS LAB_07_0124_firewall.csv', index=False, encoding='utf-8-sig')

del _predict_inverse
del concat_data
del test_data_0124

print(" 0124 Finish!")

# prediction for test_0125
print("//*******************************************//")
print("//*      Now predict for test data 0125     *//")
print("//*               Now using...              *//")
print("//*    7XGB + 4RFC + RandomForestClassifier *//")
print("//*******************************************//")
file_path = './transformed/0125_transformed_solo.csv'
test_data_0125 = pd.read_csv(file_path)

pred_prob_0125 = main(test_data_0125)

_predict = model_s2.predict(pred_prob_0125)
print("shape of _predict : ", _predict.shape)
unique, counts = np.unique(_predict, return_counts=True)
print("_predict contains:{}".format(dict(zip(unique, counts))))
print("------------------------------------------------")
print("//    Now doing label encode inverse    //")
_predict_inverse = dum_inverse(_predict)
print("shape of _predict_inverse : ", _predict_inverse.shape)
unique, counts = np.unique(_predict_inverse, return_counts=True)
print("_predict_inverse contains:{}".format(dict(zip(unique, counts))))

del _predict
print("------------------------------------------------")
concat_data = concat(test_data_0125, _predict_inverse)
print("shape of concat_data" , concat_data.shape)

print("//    Now Store the result    //")
#store to csv files 
concat_data.to_csv('./131_OASIS LAB_07_0125_firewall.csv', index=False, encoding='utf-8-sig')

del _predict_inverse
del concat_data
del test_data_0125

print(" 0125 Finish!")


# prediction for test_0126
print("//*******************************************//")
print("//*      Now predict for test data 0126     *//")
print("//*               Now using...              *//")
print("//*    7XGB + 4RFC + RandomForestClassifier *//")
print("//*******************************************//")
file_path = './transformed/0126_transformed_solo.csv'
test_data_0126 = pd.read_csv(file_path)

pred_prob_0126 = main(test_data_0126)

_predict = model_s2.predict(pred_prob_0126)
print("shape of _predict : ", _predict.shape)
unique, counts = np.unique(_predict, return_counts=True)
print("_predict contains:{}".format(dict(zip(unique, counts))))
print("------------------------------------------------")
print("//    Now doing label encode inverse    //")
_predict_inverse = dum_inverse(_predict)
print("shape of _predict_inverse : ", _predict_inverse.shape)
unique, counts = np.unique(_predict_inverse, return_counts=True)
print("_predict_inverse contains:{}".format(dict(zip(unique, counts))))

del _predict

concat_data = concat(test_data_0126, _predict_inverse)
print("shape of concat_data" , concat_data.shape)
print("------------------------------------------------")
print("//    Now Store the result    //")
#store to csv files 
concat_data.to_csv('./131_OASIS LAB_07_0126_firewall.csv', index=False, encoding='utf-8-sig')

del _predict_inverse
del concat_data
del test_data_0126
del model1, model2, model3, model4, model5, model6, model7 
del model8, model9, model10, model11

print(" 0126 Finish!")
