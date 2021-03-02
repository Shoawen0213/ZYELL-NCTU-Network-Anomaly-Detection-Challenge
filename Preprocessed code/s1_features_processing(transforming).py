#!/usr/bin/env python
# coding: utf-8

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

    print(" test_data_0123 info ")
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



print("now in ...")
now_path = os.getcwd()
file_path = os.getcwd()+'/training_set'
print(file_path)
print("//----------------------------------//")





print("Start to sort training dataset...")
path_list = os.listdir(file_path)
path_list.sort() #sort in path
print(path_list)

#concat file to one csv
print("Training data concat into one .csv ")
extension = '.csv'
all_csv_sorted = list(filter(lambda x: x[-4:]==extension , path_list))
all_csv_sorted #list

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_csv_sorted])

#export to csv
combined_csv.to_csv("combine_training_dataset.csv", index=False, encoding='utf-8-sig')


# In[ ]:


print("shape of combined_csv.csv is {}".format(combined_csv.shape))


# In[ ]:


print(combined_csv['time'][0])


combined_csv = time_src_dst_transdform(combined_csv, 1, 3, 5)

loading_print('store into csv files')

# #store into csv
combined_csv.to_csv("./train_data_transformed.csv", index=False, encoding='utf-8-sig')
print("Finish storing...")

