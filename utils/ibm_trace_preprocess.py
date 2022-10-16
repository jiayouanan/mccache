# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 17:08:08 2022

@author: lenovo
"""

import pandas as pd
import numpy as np
from numpy.random import default_rng


def split_operation(operation):
    return operation.split(".")[1]

def filter_low_key(data_df, filter_num):
    appnum_df = data_df.key.value_counts().to_frame(name='num')
    appnum_df['key'] = appnum_df.index
    # according to the data, we should first keep the hot item in result
    appnum1_df = appnum_df[appnum_df.num >= filter_num+1]
    appnum2_df = appnum_df[appnum_df.num >= filter_num]
    # since appnum2_df.shape[0] is large, we filter some items
    if appnum2_df.shape[0]>1800:
        # each sample result remains same
        appnum2_df = appnum_df.sample(n=1800,random_state=1)
        
    valid_key_list = list(appnum1_df['key'])
    appnum2_list = list(appnum2_df['key'])
    valid_key_list.extend(appnum2_list)
    data_df = data_df[data_df.key.isin(valid_key_list)]
    
    return data_df

data_df = pd.read_csv('data/ibm/ibm1_1000000.csv', names=['timestamp','operation','key'])

# filter some columns
data_df = data_df.drop(['timestamp'], axis=1) 
#data_df = data_df.astype(str)

data_df["operation"] = data_df["operation"].apply(split_operation)

# filter some rows
data_df['operation'] = data_df['operation'].replace(['DELETE','COPY','PUT'],'set')
data_df['operation'] = data_df['operation'].replace(['HEAD','GET'],'get')
valid_ope_list = ['get','set']
data_df = data_df[data_df.operation.isin(valid_ope_list)]
# we set 100 here, i.e. around 50%
wrt_freq = data_df.operation.value_counts().loc['set'] / (data_df.operation.value_counts().loc['set']+
                                         data_df.operation.value_counts().loc['get'])*100

# fix it to 12                                         
filter_num = 12
data_df = filter_low_key(data_df, filter_num).reset_index(drop=True)


# hard code encode
keymap = {elem:index for index,elem in enumerate(set(data_df["key"]))}
data_df['key'] = data_df['key'].map(keymap)


rsize = 12
wsize = 12
txn_seq_len = 5000
txn_id_seq = np.zeros(txn_seq_len, dtype=int)

rng = default_rng(624)
flag_arr = rng.random(txn_seq_len)
write_flag_arr = flag_arr < wrt_freq

txn_tuple_dict = {} 

# item_size_dict and cls_item_dict are useless. just make sure all the datasets in a unified format.
item_size_dict = {}
cls_item_dict = {}

write_flag_list = []
qry_ix = 0
unique_qry_num = len(data_df.key.value_counts())


txn_item_dict={}
for txn_idx in range(txn_seq_len):
    print("txn_idx", txn_idx)
    txn_size = wsize if write_flag_arr[txn_idx] else rsize
    txn_qid_list= []
    while len(txn_qid_list) < txn_size:
        tmp_key = data_df['key'][qry_ix]
        if tmp_key in txn_qid_list:
            # here we allow the duplicate keys in a txn by the prob, which is different from wiki and twit.
            if rng.random(1)>0.05:
                txn_qid_list.append(tmp_key)
            qry_ix += 1
            continue
        txn_qid_list.append(tmp_key)
        qry_ix += 1
    txn_tuple_dict[txn_idx] = txn_qid_list

    # generate txn_item_dict based on txn_tuple_dict
    for k,v in txn_tuple_dict.items():
        tmp_txn_item_arr = np.zeros(unique_qry_num, dtype=bool)
        for i in v:
            tmp_txn_item_arr[i] = 1
        txn_item_dict[k] = tmp_txn_item_arr
    
    item_size_dict[txn_idx] = 100
    txn_id_seq[txn_idx] = txn_idx

cls_item_dict[0] = np.zeros(unique_qry_num, dtype=bool)+1

for item_id in range(unique_qry_num):
    item_size_dict[item_id] = 100


from pathlib import Path
import pickle

workload_dir = "d:/stream_cc_exp/data/ibm/Ibm_RSize{}_WSize{}_Len{}".format(rsize, wsize, txn_seq_len)
Path(workload_dir).mkdir(parents=True, exist_ok=True)
np.save(workload_dir + '/flag_seq.npy', write_flag_arr)
np.save(workload_dir + '/id_seq.npy', txn_id_seq)
with open(workload_dir + '/item_size.pkl', 'wb') as item_size_fp:
    pickle.dump(item_size_dict, item_size_fp)
with open(workload_dir + '/cls_item.pkl', 'wb') as cls_item_fp:
    pickle.dump(cls_item_dict, cls_item_fp)
with open(workload_dir + '/txn_item.pkl', 'wb') as txn_item_fp:
    pickle.dump(txn_item_dict, txn_item_fp)

print('Total Transactions: {}, Read Transactions: {}, Write Transactions: {}'.format(txn_seq_len, txn_seq_len - sum(write_flag_arr), sum(write_flag_arr)))
print('unique query number: {}'.format(unique_qry_num))
print('unique transaction number: {}'.format(txn_seq_len))


aa = data_df.key.value_counts()
data_df.info()