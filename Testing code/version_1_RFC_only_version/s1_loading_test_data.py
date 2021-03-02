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


# In[6]:


def ip2int(addr):
    return struct.unpack("!I", socket.inet_aton(addr))[0]
    
def time_form_change(timeString):
    struct_time = time.strptime(timeString, "%Y-%m-%d %H:%M:%S") # 轉成時間元組
    time_stamp = int(time.mktime(struct_time)) # 轉成時間戳
    return time_stamp

def loading_print(string):
    prt_str = string
    print("{}".format(prt_str), end = "")
    for i in range(10):
        print(".", end = '',flush = True)
        time.sleep(0.2)
    print('\n')

def time_src_dst_transdform(dataframe, insert_pos1, insert_pos2, insert_pos3):
    data_len = len(dataframe)
    loading_print('Adding columns for int_time')
    loading_print('Adding columns for int_src, int_dst')
    
    list_int_time = [0] * data_len
    list_int_src  = [0] * data_len
    list_int_dst  = [0] * data_len
    
    dataframe.insert(insert_pos1, "int_time", list_int_time, True)
    dataframe.insert(insert_pos2, "int_src" , list_int_src,  True)
    dataframe.insert(insert_pos3, "int_dst" , list_int_dst,  True)
    
    print(" << Finish adding >>")
    print('\n')

    print(" test_data info ")
    print('\n')
    dataframe.info()
    
    print("Start to transform time & IPv4 into integer...")
    
    k=1
    for i in tqdm(range(data_len)):
        if(i%10000==0)and(i>0):
            print('The', k ,'th part(per 10000) of data')
            k=k+1
            print(i)
        dataframe["int_time"][i] = (time_form_change(dataframe["time"][i]))
        dataframe["int_src"][i]  = (ip2int(dataframe["src"][i]))
        dataframe["int_dst"][i]  = (ip2int(dataframe["dst"][i]))
        sleep(0.01)
        
    print("Finish trainsforming...")   
    return dataframe 


# In[4]:


print("now move to ...")
now_path = os.getcwd()
file_path = os.getcwd()+ '/testing_dataset'
print(file_path)


# In[5]:


test_data_path = file_path = './0123_firewall.csv'
test_data_0123 = pd.read_csv(test_data_path)

print("shape of 0123_firewall.csv is {}".format(test_data_0123.shape)) # 3601186 rows × 22 columns
print(test_data_0123['time'][0])


# In[7]:


test_data_0123 = time_src_dst_transdform(test_data_0123, 1, 3, 5)

loading_print('store into csv files')

# #store into csv
test_data_0123.to_csv("./transformed/0123_transformed_solo.csv", index=False, encoding='utf-8-sig')
print("Finish storing...")


# In[8]:


test_data_path = file_path = './0124_firewall.csv'
test_data_0124 = pd.read_csv(test_data_path)

print("shape of 0124_firewall.csv is {}".format(test_data_0124.shape)) # 2050710 rows × 22 columns


# In[ ]:


test_data_0124 = time_src_dst_transdform(test_data_0124, 1, 3, 5)

loading_print('store into csv files')

# #store into csv
test_data_0124.to_csv("./transformed/0124_transformed_solo.csv", index=False, encoding='utf-8-sig')
print("Finish storing...")


# In[9]:


test_data_path = file_path = './0125_firewall.csv'
test_data_0125 = pd.read_csv(test_data_path)

print("shape of 0125_firewall.csv is {}".format(test_data_0125.shape)) # 2120819 rows × 22 columns


# In[ ]:


test_data_0125 = time_src_dst_transdform(test_data_0125, 1, 3, 5)

loading_print('store into csv files')

# #store into csv
test_data_0125.to_csv("./transformed/0125_transformed_solo.csv", index=False, encoding='utf-8-sig')
print("Finish storing...")


# In[10]:


test_data_path = file_path = './0126_firewall.csv'
test_data_0126 = pd.read_csv(test_data_path)

print("shape of 0126_firewall.csv is {}".format(test_data_0126.shape)) # 5517815 rows × 22 columns


# In[ ]:


test_data_0126 = time_src_dst_transdform(test_data_0126, 1, 3, 5)

loading_print('store into csv files')

# #store into csv
test_data_0126.to_csv("./transformed/0126_transformed_solo.csv", index=False, encoding='utf-8-sig')
print("Finish storing...")

