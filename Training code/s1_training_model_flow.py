import os
import glob
import sys
import time
import datacompy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance

def train_model(model_algorithm, dataframe, split, random_state, learning_rate, seed, max_depth):
    SPLIT = split
    RAND_SEED = random_state
    dataframe_dum_app = dataframe.join(pd.get_dummies(dataframe.app))
    dataframe_dum_app = dataframe_dum_app.drop(columns = ['time', 'src', 'dst', 'app'])
    dataframe_dum_app #1848289 rows × 63 columns

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
    if compare.df2_unq_columns() == str(set)+str(()):
        print("There;s just completely the same.")
    else :
        print("{} is missing".format(compare.df2_unq_columns()))
    print("//----------------------------------------------//")

    x = dataframe_66col[['int_time', 'int_src', 'int_dst', 'spt', 'dpt', 'duration',
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
       'tftp', 'vdolive']]  #66
    y = dataframe_66col[['label']]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=SPLIT, random_state=RAND_SEED)

    # use label encode on label

    '''
          Argument:
              'DDOS-smurf'         --> 0
              'Normal'             --> 1
              'Probing-Nmap'       --> 2
              'Probing-Port sweep' --> 3
              'Probing-IP sweep'   --> 4
        '''
    label_list = ['Normal', 'DOS-smurf', 'Probing-Ipsweep', 'Probing-Nmap', 'Probing-Port sweep']
    
    labelencoder = LabelEncoder()
    y_train['label'] = labelencoder.fit_transform(y_train['label'])
    y_test['label']  = labelencoder.fit_transform(y_test['label'])

    unique, counts = np.unique(y_train['label'], return_counts=True)
    print("y_train['label'] contains:{} ".format(dict(zip(unique, counts))))

    unique, counts = np.unique(y_test['label'], return_counts=True)
    print("y_test['label'] contains:{} ".format(dict(zip(unique, counts))))

    print("x_train shape is: {}".format(x_train.shape))
    print("x_test shape  is: {}".format(x_test.shape))
    print("y_train shape is: {}".format(y_train.shape))
    print("y_test shape  is: {}".format(y_test.shape))
    
    #choose model algorithm 
    if model_algorithm == 0 :
        model = XGBClassifier()
        model.fit(x_train, y_train)
        print('The accuracy of eXtreme Gradient Boosting Classifier on testing set:', model.score(x_test, y_test))
    else :
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        print('The accuracy of eXtreme Gradient Boosting Classifier on testing set:', model.score(x_test, y_test))

    return model, x_test, y_test

def valid_modell(model, x_test, y_test):
    _predict = model.predict(x_test)
    print("shape of _predict : ", _predict.shape)
    unique, counts = np.unique(_predict, return_counts=True)
    print("_predict contains:{}".format(dict(zip(unique, counts))))

    print('The accuracy of XGBClassifier on testing data set', model.score(x_test, y_test))

    #calculate score by function wrote by us
    %run score.py

    #transform into numpy array
    y_array = np.array(y_test)
    _pred_array = np.array(_predict).reshape(len(_predict), 1)

    score_ = score(_pred_array, y_array)
    print('The score that calculated by SP version of XGBClassifier on testing data set', score_)
    
    return y_array, _pred_array
    


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

file_path = ''' fill training datasets locations here'''
train_data_ = pd.read_csv(file_path)
train_data_     

#(model_algorithm, dataframe, split, random_state, learning_rate, seed, max_depth)
'''
    model_algorithm = '0' -->  XGBClassifier()
    model_algorithm = '1' -->  RandomForestClassifier()
'''
model, x_test, y_test = train_model(0, train_data_, 0.25, 60, 0.05, 123, 6)  #defined by yourself

#calculate score by function wrote by us & score function by API
y_array, _pred_array = valid_modell(model, x_test, y_test)

#save model
joblib.dump(model, ''' fill locations you want to stroe''')
