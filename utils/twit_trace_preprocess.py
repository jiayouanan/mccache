# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 19:42:47 2022

@author: lenovo
"""

import pandas as pd
import numpy as np

def split_key(key):
    return key.split("-")[0]

def filter_low_key(data_df, filter_num):
    appnum_df = data_df.key.value_counts().to_frame(name='num')
    appnum_df['key'] = appnum_df.index
    appnum_df = appnum_df[appnum_df.num >= filter_num]
    
    valid_key_list = list(appnum_df['key'])
    data_df = data_df[data_df.key.isin(valid_key_list)]
    
    return data_df

def preprocess(data_df):
    # filter some columns
    data_df = data_df.drop(['timestamp', 'key_size','value_size','client','TTL'], axis=1) 
    #data_df = data_df.astype(str)
    
    # filter some rows
    data_df['operation'] = data_df['operation'].replace(['delete'],'set')
    valid_ope_list = ['get','set']
    data_df = data_df[data_df.operation.isin(valid_ope_list)]
    # 1 means write. 0 means read.
    data_df['operation'] = data_df['operation'].replace(['set'],1)
    data_df['operation'] = data_df['operation'].replace(['get'],0)
    # Note different cluster trace may have different dilimits.
    data_df["key"] = data_df["key"].apply(split_key)
    
    # filter those keys whose appearance less than a constant
    filter_num = 5
    data_df = filter_low_key(data_df, filter_num)
    
    # hard code encode
    keymap = {elem:index for index,elem in enumerate(set(data_df["key"]))}
    data_df['key'] = data_df['key'].map(keymap)
    
    return data_df



data_df = pd.read_csv('data/twit/twit1_50000.csv', names=['timestamp','key','key_size','value_size','client'
                                               ,'operation','TTL'])
data_df = preprocess(data_df)     
# DataFrame Group by Consecutive Same String Values
consecutives = data_df['operation'].diff().ne(0).cumsum()
key_txn = data_df['key'].groupby(consecutives).agg(list)
wrt_flag_txn = data_df['operation'].groupby(consecutives).agg(list)

rsize = 3
wsize = 2
txn_seq_len = 5000
txn_id_seq = np.zeros(txn_seq_len, dtype=int)

txn_item_dict1 = {}
txn_tuple_dict = {} 

# item_size_dict and cls_item_dict are useless. just make sure all the datasets in a unified format.
item_size_dict = {}
cls_item_dict = {}

write_flag_list = []
unique_txn_num = 0
unique_qry_num = len(data_df.key.value_counts())
for txn_id in range(len(key_txn)):
    print(txn_id)
    tmp_key_set = set(key_txn.iloc[txn_id])
    tmp_wrt_flag_txn = set(wrt_flag_txn.iloc[txn_id]) 
    # if it is a read and the length of txn is less than 3, continue
    if list(tmp_wrt_flag_txn)[0] == 0 and len(tmp_key_set)<rsize:
        continue
    # if it is a wrt and the length of txn is less than 2, continue
    if list(tmp_wrt_flag_txn)[0] == 1 and len(tmp_key_set)<wsize:
        continue
    
    txn_tuple_dict[unique_txn_num] = list(tmp_key_set)
    write_flag_list.append(True if list(tmp_wrt_flag_txn)[0] == 1 else False )
#    unique_qry_num.update(tmp_key_set)
    
    # generate txn_item_dict based on txn_tuple_dict
    for k,v in txn_tuple_dict.items():
        tmp_txn_item_arr = np.zeros(unique_qry_num, dtype=bool)
        for i in v:
            tmp_txn_item_arr[i] = 1
        txn_item_dict1[k] = tmp_txn_item_arr
        
    txn_id_seq[unique_txn_num] = unique_txn_num
    unique_txn_num += 1
    if unique_txn_num==5000:
        break

for item_id in range(unique_qry_num):
    item_size_dict[item_id] = 100
       
write_flag_arr = np.array(write_flag_list)
cls_item_dict[0] = np.zeros(unique_qry_num, dtype=bool)+1
 
from pathlib import Path
import pickle

workload_dir = "d:/stream_cc_exp/data/twit/Twit_RSize{}_WSize{}_Len{}".format(rsize, wsize, txn_seq_len)
Path(workload_dir).mkdir(parents=True, exist_ok=True)
np.save(workload_dir + '/flag_seq.npy', write_flag_arr)
np.save(workload_dir + '/id_seq.npy', txn_id_seq)
with open(workload_dir + '/item_size.pkl', 'wb') as item_size_fp:
    pickle.dump(item_size_dict, item_size_fp)
with open(workload_dir + '/cls_item.pkl', 'wb') as cls_item_fp:
    pickle.dump(cls_item_dict, cls_item_fp)
with open(workload_dir + '/txn_item.pkl', 'wb') as txn_item_fp:
    pickle.dump(txn_item_dict1, txn_item_fp)

print('Total Transactions: {}, Read Transactions: {}, Write Transactions: {}'.format(txn_seq_len, txn_seq_len - sum(write_flag_arr), sum(write_flag_arr)))
print('unique query number: {}'.format(unique_qry_num))
print('unique transaction number: {}'.format(unique_txn_num))

# aa = data_df.key.value_counts()
# data_df.info()

