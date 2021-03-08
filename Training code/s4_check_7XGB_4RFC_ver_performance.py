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

import sys
sys.path.append('./score_SP.py')
import score_SP


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

def main(dataframe):
    print('//---------------------//')
    print("// now using model1... //")
    print('//---------------------//')
    model1_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model1)
        
    #model1_pred_df = pd.DataFrame(model1_pred)
    model1_pred_prob_df = pd.DataFrame(model1_pred_prob)
    
    #model1_pred_df.to_csv('./training_data_model1_pred.csv', index=False, encoding='utf-8-sig')
    #model1_pred_prob_df.to_csv('./training_data_model1_pred_prob.csv', index=False, encoding='utf-8-sig')
    
    #del model1_pred
    del model1_pred_prob
    
    print('//---------------------//')
    print("// now using model2... //")
    print('//---------------------//')
    model2_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model2)
   
    #model2_pred_df = pd.DataFrame(model2_pred)
    model2_pred_prob_df = pd.DataFrame(model2_pred_prob)
   
    #model2_pred_df.to_csv('./training_data_model2_pred.csv', index=False, encoding='utf-8-sig')
    #model2_pred_prob_df.to_csv('./training_data_model2_pred_prob.csv', index=False, encoding='utf-8-sig')
     
    #del model2_pred
    del model2_pred_prob
   
    print('//---------------------//')
    print("// now using model3... //")
    print('//---------------------//')
    model3_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model3)
   
    #model3_pred_df = pd.DataFrame(model3_pred)
    model3_pred_prob_df = pd.DataFrame(model3_pred_prob)
   
    #model3_pred_df.to_csv('./training_data_model3_pred.csv', index=False, encoding='utf-8-sig')
    #model3_pred_prob_df.to_csv('./training_data_model3_pred_prob.csv', index=False, encoding='utf-8-sig')
     
    #del model3_pred
    del model3_pred_prob

    print('//---------------------//')
    print("// now using model4... //")
    print('//---------------------//')
    model4_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model4)
   
    #model4_pred_df = pd.DataFrame(model4_pred)
    model4_pred_prob_df = pd.DataFrame(model4_pred_prob)
   
    #model4_pred_df.to_csv('./training_data_model4_pred.csv', index=False, encoding='utf-8-sig')
    #model4_pred_prob_df.to_csv('./training_data_model4_pred_prob.csv', index=False, encoding='utf-8-sig')
     
    #del model4_pred
    del model4_pred_prob
     
    print('//---------------------//')
    print("// now using model5... //")
    print('//---------------------//')
    model5_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model5)
   
    #model5_pred_df = pd.DataFrame(model5_pred)
    model5_pred_prob_df = pd.DataFrame(model5_pred_prob)
   
    #model5_pred_df.to_csv('./training_data_model5_pred.csv', index=False, encoding='utf-8-sig')
    #model5_pred_prob_df.to_csv('./training_data_model5_pred_prob.csv', index=False, encoding='utf-8-sig')
     
    #del model5_pred
    del model5_pred_prob
     
     
    print('//---------------------//')
    print("// now using model6... //")
    print('//---------------------//')
    model6_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model6)
   
    #model6_pred_df = pd.DataFrame(model6_pred)
    model6_pred_prob_df = pd.DataFrame(model6_pred_prob)
   
    #model6_pred_df.to_csv('./training_data_model6_pred.csv', index=False, encoding='utf-8-sig')
    #model6_pred_prob_df.to_csv('./training_data_model6_pred_prob.csv', index=False, encoding='utf-8-sig')
     
    #del model6_pred
    del model6_pred_prob
     
    
    print('//---------------------//')
    print("// now using model7... //")
    print('//---------------------//')
    model7_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model7)
   
    #model7_pred_df = pd.DataFrame(model7_pred)
    model7_pred_prob_df = pd.DataFrame(model7_pred_prob)
   
    #model7_pred_df.to_csv('./training_data_model7_pred.csv', index=False, encoding='utf-8-sig')
    #model7_pred_prob_df.to_csv('./training_data_model7_pred_prob.csv', index=False, encoding='utf-8-sig')
     
    #del model7_pred
    del model7_pred_prob
    
    print('//---------------------//')
    print("// now using model8... //")
    print('//---------------------//')
    model8_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model8)
   
    #model8_pred_df = pd.DataFrame(model8_pred)
    model8_pred_prob_df = pd.DataFrame(model8_pred_prob)
   
    #model8_pred_df.to_csv('./training_data_model8_pred.csv', index=False, encoding='utf-8-sig')
    #model8_pred_prob_df.to_csv('./training_data_model8_pred_prob.csv', index=False, encoding='utf-8-sig')
     
    #del model8_pred
    del model8_pred_prob
    
    print('//---------------------//')
    print("// now using model9... //")
    print('//---------------------//')
    model9_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model9)
   
    #model9_pred_df = pd.DataFrame(model9_pred)
    model9_pred_prob_df = pd.DataFrame(model9_pred_prob)
   
    #model9_pred_df.to_csv('./training_data_model9_pred.csv', index=False, encoding='utf-8-sig')
    #model9_pred_prob_9df.to_csv('./training_data_model9_pred_prob.csv', index=False, encoding='utf-8-sig')
     
    #del model9_pred
    del model9_pred_prob
    
    print('//---------------------//')
    print("// now using model10... //")
    print('//---------------------//')
    model10_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model10)
   
    #model10_pred_df = pd.DataFrame(model10_pred)
    model10_pred_prob_df = pd.DataFrame(model10_pred_prob)
   
    #model10_pred_df.to_csv('./training_data_model10_pred.csv', index=False, encoding='utf-8-sig')
    #model10_pred_prob_9df.to_csv('./training_data_model10_pred_prob.csv', index=False, encoding='utf-8-sig')
     
    #del model10_pred
    del model10_pred_prob
    
    print('//---------------------//')
    print("// now using model11... //")
    print('//---------------------//')
    model11_pred_prob = fit_data_dum(dataframe, org_list,  org_df, model11)
   
    #model11_pred_df = pd.DataFrame(model11_pred)
    model11_pred_prob_df = pd.DataFrame(model11_pred_prob)
   
    #model11_pred_df.to_csv('./training_data_model11_pred.csv', index=False, encoding='utf-8-sig')
    #model11_pred_prob_9df.to_csv('./training_data_model11_pred_prob.csv', index=False, encoding='utf-8-sig')
     
    #del model11_pred
    del model11_pred_prob

    
    print('//-------------------------//')
    print("// concat all predict data //")
    print('//-------------------------//')
    #concat_pred_df = pd.concat([model1_pred_df, model2_pred_df, model3_pred_df, model4_pred_df, model5_pred_df, model6_pred_df,
    #                           model7_pred_df, model8_pred_df, model9_pred_df, model10_pred_df, model11_pred_df], axis=1)
   
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
    del dataframe_66col_reload
    
    return model_pred_prob
    
def valid_modell(model, x_test, y_test):
    _predict = model.predict(x_test)
    print("shape of _predict : ", _predict.shape)
    unique, counts = np.unique(_predict, return_counts=True)
    print("_predict contains:{}".format(dict(zip(unique, counts))))

    print('The accuracy on testing data set', model.score(x_test, y_test))

    #transform into numpy array
    y_array = np.array(y_test)
    _pred_array = np.array(_predict).reshape(len(_predict), 1)

    score_ = score_SP.score(_pred_array, y_array)
    print('The score that calculated by SP version is', score_)



print('//---------------//')
print('// NOW Fitting   //')
print('//---------------//')

file_path = '/home/shaowen0213/NAD/Time&&IP_in_int(26col).csv'
train_data_transformed = pd.read_csv(file_path)

x = train_data_transformed.drop(columns = ['label'])
y = train_data_transformed['label']

'''
  Argument:
      'Normal'             --> 1
      'Probing-Nmap'       --> 3
      'Probing-Port sweep' --> 4
      'Probing-IP sweep'   --> 2
      'DDOS-smurf'         --> 0
'''
print("//**************************//")
print("//*    label encode on y   *//")
print("//**************************//")
label_list = ['Normal', 'DOS-smurf', 'Probing-Ipsweep', 'Probing-Nmap', 'Probing-Port sweep']
print("label_list : {}".format(label_list))
print("//--------------------------------------------//")

#label encode
labelencoder = LabelEncoder()
y_LE = labelencoder.fit_transform(y)
print("y_train labeled after label encode : {}".format(y_LE))

print("shape of y_LE data : ", y_LE.shape)
unique, counts = np.unique(y_LE, return_counts=True)
print("y_LE data contains:{} ".format(dict(zip(unique, counts))))

print("//--------------------------------------------//")

'''
y.to_csv('./training_data_y_label.csv', index=False, encoding='utf-8-sig')
print("shape of predict data : ", y.shape)
'''
del y
del train_data_transformed


concat_pred_prob_df = main(x)
print("shape of concat_pred_prob_df", concat_pred_prob_df.shape)

#print("storing concat_pred_df ")
#concat_pred_df.to_csv('./training_data_X_pred.csv', index=False, encoding='utf-8-sig')

'''
print("storing concat_pred_prob ")
#store to csv files 
concat_pred_prob_df.to_csv('./training_data_X_pred_prob_7RFC_ver2.csv', index=False, encoding='utf-8-sig')

del concat_pred_prob_df
'''

print("Loading stage 2 model...")
m=0
test_model1 = joblib.load('./0306_final_stage_dt_clfver_7XGB_ver2.pkl')
m+=1
test_model2 = joblib.load('./0306_final_stage_mlp_clfver_7XGB_ver2.pkl')
m+=1
test_model3 = joblib.load('./0306_final_stage_rf_clfver_7XGB_ver2.pkl')
m+=1
print("Loading {} models Successes" .format(m))

print("//*******************************************//")
print("//*    Start to predict the final result    *//")
print("//*               Now using...              *//")
print("//*    7RFC + 4XGB + DecisionTreeClassifier *//")
print("//*******************************************//")
valid_modell(test_model1, concat_pred_prob_df, y_LE)
del test_model1
print("**********************************************************")

print("//*******************************************//")
print("//*    Start to predict the final result    *//")
print("//*               Now using...              *//")
print("//*    7RFC + 4XGB + MLPClassifier *//")
print("//*******************************************//")
valid_modell(test_model2, concat_pred_prob_df, y_LE)
del test_model2
print("**********************************************************")

print("//*******************************************//")
print("//*    Start to predict the final result    *//")
print("//*               Now using...              *//")
print("//*    7RFC + 4XGB + RandomForestClassifier *//")
print("//*******************************************//")
valid_modell(test_model3, concat_pred_prob_df, y_LE)
del test_model3
print("**********************************************************")

del concat_pred_prob_df
del y_LE
print("Finish !!")


