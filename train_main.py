# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 10:16:42 2022

@author: lenovo
"""

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
from collections import defaultdict
from ML.lgbm import LgbmPreClass
from ML.feature import generate_read_features
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from utils.load_dataset import load_item_univ, load_txn_univ, load_txn_seq, load_ycsb_seq
import configparser
import yaml
from MCCache.cache_alg import SingleAlg
from sklearn.metrics import accuracy_score

pd.options.mode.chained_assignment = None


#####
# read data set
####

def alg_config_parse(filename='config/sysTest.yaml'):
    with open(filename, 'r') as fp:
        try:
            alg_dict = yaml.safe_load(fp)
            alg_dict['dataset_dir'] = alg_dict['dataset_root'] + "/" + alg_dict['dataset_name']
            return alg_dict
        except yaml.YAMLError as exc:
            print(exc)

alg_dict = alg_config_parse('config/sysTest.yaml')


#dataset_dir='ML/'+alg_dict['dataset_dir']
dataset_dir = alg_dict['dataset_dir']
csize=alg_dict['csize']
cache_scheme=alg_dict['cache_scheme']
alg_name=alg_dict['alg_name']
batch_size=alg_dict['batch_size']
opt_len=alg_dict['opt_len']
fetch_strategy=alg_dict['fetch_strategy']
staleness_bound=alg_dict['staleness_bound']
ML_model_root = alg_dict['ML_model_root']
ob_acc = alg_dict['ob_acc']
semionline_flag = alg_dict['semionline_flag']


item_size_path = dataset_dir + '/item_size.pkl'
cls_item_path = dataset_dir + '/cls_item.pkl'
txn_item_path = dataset_dir + '/txn_item.pkl'
id_seq_path = dataset_dir + '/id_seq.npy'
flag_seq_path = dataset_dir + '/flag_seq.npy'
ycsb_seq_path = dataset_dir + '/transactions.dat'

item_size_dict, cls_item_dict = load_item_univ(item_size_path, cls_item_path)
txn_item_dict = load_txn_univ(txn_item_path)
txn_id_seq, write_flag_seq = load_txn_seq(id_seq_path, flag_seq_path)

query_num = len(item_size_dict)
cls_num = len(cls_item_dict)
seq_len = len(txn_id_seq)
read_qry_cnt, wrt_qry_cnt = 0, 0
wrt_txn_cnt = write_flag_seq.sum()
read_txn_cnt = len(write_flag_seq) - wrt_txn_cnt
item_read_time_dict, item_write_time_dict = {i:[] for i in range(query_num)}, {i:[] for i in range(query_num)}
for time_step in range(len(txn_id_seq)):
    txn_vec = txn_item_dict[txn_id_seq[time_step]]
    if write_flag_seq[time_step]:
        wrt_qry_cnt += np.sum(txn_vec)
        for item_id in np.where(txn_vec == 1)[0]:
            item_write_time_dict[item_id].append(time_step)
    else:
        read_qry_cnt += np.sum(txn_vec)
        for item_id in np.where(txn_vec == 1)[0]:
            item_read_time_dict[item_id].append(time_step)
dist_read_qry_cnt, dist_wrt_qry_cnt = 0, 0
for i in range(query_num):
    if len(item_read_time_dict[i]) > 0:
        dist_read_qry_cnt += 1
    if len(item_write_time_dict[i]) > 0:
        dist_wrt_qry_cnt += 1
total_item_size = sum(item_size_dict.values())

workload_stats = {'query_num': query_num, 'cls_num': cls_num, 
            'total_size': total_item_size, 'seq_len': seq_len,
            'read_txn_cnt': read_txn_cnt, 'write_txn_cnt': wrt_txn_cnt, 
            'read_qry_cnt': read_qry_cnt, 'write_qry_cnt': wrt_qry_cnt, 
            'unique_read_qry_cnt': dist_read_qry_cnt, 'unique_write_qry_cnt': dist_wrt_qry_cnt}

print(workload_stats)

#####
# generate training set
####

def batch_step_process(time_step: int, batch_start: int, batch_end: int, txn_id_seq: np.ndarray, 
        write_flag_seq: np.ndarray, item_size_dict: dict, txn_item_dict: dict, cls_item_dict: dict, mark_no_read=False):
        """ Process one transaction in a batch. 
        
            Note that txn_id = time_step
        """
        txn_id = txn_id_seq[time_step - batch_start]
        txn_vec = txn_item_dict[txn_id]
        
        ob_item_vec = np.zeros(query_num, dtype=bool)
        # handle write transaction
        if write_flag_seq[time_step - batch_start]:
            wrt_rt_val = alg_obj.cache.proc_write_txn(txn_id=txn_id, txn_vec=txn_vec, time_step=time_step, cache_scheme=alg_obj.cache_scheme, txn_alg=alg_obj.txn_alg)
        # handle read transaction
        else:
            cch_check, max_ver = alg_obj.cache.proc_read_txn(txn_id=txn_id, txn_vec=txn_vec, time_step=time_step)
            
            # detect and evict obsolete queries for OFF algorithm
            if 'OFF' in alg_name or 'oMCP' in alg_name or 'sMCP' in alg_name:
                # Actually, np.sum(ob_item_vec)=0
                ob_item_vec = alg_obj.cache.findOB(time_step=time_step, txn_item_dict=txn_item_dict, txn_id_seq=txn_id_seq, write_flag_seq=write_flag_seq, alg_name=alg_name, trunc_len=alg_obj.findOB_trunc, outside_init=False, mark_no_read=mark_no_read, batch_start=batch_start) 
                
                alg_obj.ob_cnt += np.sum(ob_item_vec)
                alg_obj.cache.evict_from_cache(txn_id=0, item_vec=ob_item_vec, txn_alg=False)
            
            # First find OB, then check CCH 
            if cch_check:   # jump to next transaction if CCH
                alg_obj.cch_cnt += 1
                return 0, ob_item_vec  # execute read transaction using cache upon CCH
            
            # check if cache miss and further eviction is needed
            miss_flag, evict_size, evict_candidates = alg_obj.cache.check_miss(cache_size=alg_obj.cache_size, txn_vec=txn_vec, item_size_dict=item_size_dict)
            if miss_flag:
                alg_obj.miss_cnt += 1
            # do eviction based on algorithm & eviction candidates
            if evict_size > 0:
                next_arrival_time = alg_obj.get_next_read_time(evict_candidates, time_step, batch_end, trunc=-1)
                if alg_obj.alg_name == 'OFF' or alg_obj.alg_name == 'bMCP' or alg_obj.alg_name == 'sMCP':
                    alg_obj.evict_cnt += alg_obj.cache.evict_belady_dist(item_size_dict=item_size_dict, evict_candidates=evict_candidates, next_arrival_time=next_arrival_time, evict_size=evict_size)
                    #self.evict_cnt += self.cache.evict_cls_dist(item_size_dict=item_size_dict, cls_item_dict=cls_item_dict, evict_candidates=evict_candidates, next_arrival_time=next_arrival_time, evict_size=evict_size)
                elif alg_obj.alg_name == 'Belady':
                    alg_obj.evict_cnt += alg_obj.cache.evict_belady_dist(item_size_dict=item_size_dict, evict_candidates=evict_candidates, next_arrival_time=next_arrival_time, evict_size=evict_size)
                elif alg_obj.alg_name == 'LRU' or alg_obj.alg_name == 'oMCP':
                    alg_obj.evict_cnt += alg_obj.cache.evict_lru(item_size_dict=item_size_dict, evict_candidates=evict_candidates, evict_size=evict_size)
                elif alg_obj.alg_name == 'LRU_k':
                    alg_obj.evict_cnt += alg_obj.cache.evict_lru_k(item_size_dict=item_size_dict, evict_candidates=evict_candidates, evict_size=evict_size)
                elif alg_obj.alg_name == 'Belady_txn':
                    alg_obj.evict_cnt += alg_obj.cache.evict_belady_txn(evict_size=evict_size, evict_candidates=evict_candidates, item_size_dict=item_size_dict, txn_item_dict=txn_item_dict, txn_id_seq=txn_id_seq, write_flag_seq=write_flag_seq, time_step=time_step)
                else:
                    assert alg_obj.alg_name == 'LRU_txn'
                    alg_obj.evict_cnt += alg_obj.cache.evict_lru_txn(evict_size=evict_size, evict_candidates=evict_candidates, item_size_dict=item_size_dict, txn_item_dict=txn_item_dict)
            miss_cch, update_qry_num = alg_obj.cache.cmplt_read_txn(txn_id=txn_id, txn_vec=txn_vec, miss_flag=miss_flag, time_step=time_step, max_ver=max_ver, alg_name=alg_obj.alg_name, cache_scheme=alg_obj.cache_scheme)
            assert miss_cch == True
            if miss_cch:
                alg_obj.cost += update_qry_num
                alg_obj.whole_cost += update_qry_num  

            return 1, ob_item_vec
        
        return 0, ob_item_vec

def init_read_write_time(txn_id_seq: np.ndarray, write_flag_seq: np.ndarray, txn_item_dict: dict, batch_start=0) -> None:
    """Get item read time and write time initialized.

    """
    item_num = alg_obj.cache.item_num
    item_read_time_dict, item_write_time_dict = {i:[] for i in range(item_num)}, {i:[] for i in range(item_num)}
    for time_step in range(len(txn_id_seq)):
        txn_vec = txn_item_dict[txn_id_seq[time_step]]
        if write_flag_seq[time_step]:
            for item_id in np.where(txn_vec == 1)[0]:
                item_write_time_dict[item_id].append(time_step + batch_start)
        else:
            for item_id in np.where(txn_vec == 1)[0]:
                item_read_time_dict[item_id].append(time_step + batch_start)
                
    return item_read_time_dict, item_write_time_dict

cache_size =  int(csize * workload_stats['total_size'])
alg_obj = SingleAlg(alg_name, fetch_strategy)
# using opt_len as findOB truncate length for optimization
alg_obj.workload_init(cache_size, cache_scheme, item_num=workload_stats['query_num'], findOB_trunc=opt_len, staleness_bound=staleness_bound, ob_acc=ob_acc)

# we just set 500 and 2500 
batch_start, batch_end = 500, 2500


# get seq from 0 to 2500
item_read_time_dict, item_write_time_dict = init_read_write_time(txn_id_seq=txn_id_seq[0:batch_end], 
    write_flag_seq=write_flag_seq[0:batch_end], 
    txn_item_dict=txn_item_dict, batch_start=0)

# sorting queries from big to small
for i in item_read_time_dict.keys(): 
    item_read_time_dict[i].sort(reverse=True)
for i in item_write_time_dict.keys(): 
    item_write_time_dict[i].sort(reverse=True)

item_read_time_df = pd.DataFrame({'read_time_trace':list(item_read_time_dict.values())}, 
                   index=list(item_read_time_dict.keys()))

item_write_time_df = pd.DataFrame({'write_time_trace':list(item_write_time_dict.values())}, 
                   index=list(item_write_time_dict.keys()))



# get item read and write time for current batch
alg_obj.init_read_write_time(txn_id_seq=txn_id_seq[batch_start:batch_end], 
    write_flag_seq=write_flag_seq[batch_start:batch_end], 
    txn_item_dict=txn_item_dict, batch_start=batch_start)


train_feature_df = pd.DataFrame()
train_label_df = pd.DataFrame()
train_label_item_list = []
train_label_time_list = []


seq_start_time = time.time()

for time_step in range(batch_start, batch_end):
    _, ob_item_vec = batch_step_process(time_step, batch_start, batch_end, txn_id_seq[batch_start:batch_end], \
                    write_flag_seq[batch_start:batch_end], item_size_dict, txn_item_dict, cls_item_dict)
    
    # generate label
    for item_id in np.where(ob_item_vec == 1)[0]:
        train_label_item_list.append(item_id)
        # get first time_step smaller than currenr time_step
        small_time_step = list(filter(lambda i: i < time_step, item_read_time_df.loc[item_id,:][0]))[0]
        train_label_time_list.append(small_time_step)
               
    # generate feature. ONlY read txns can be used to train.
    if write_flag_seq[time_step]==False: 
        train_feature_df = train_feature_df.append(generate_read_features(time_step, batch_start, batch_end, \
                               txn_id_seq[batch_start:batch_end], txn_item_dict, item_read_time_df, item_write_time_df, semionline_flag))
   

train_label_dict = {'item_id':train_label_item_list,'time_step':train_label_time_list}
train_label_df = pd.DataFrame(train_label_dict, columns=['item_id','time_step'])
train_label_df['label'] = 1

train_feature_res_df = pd.merge(train_feature_df, train_label_df, left_on=['item_id','time_step'],right_on=['item_id','time_step'],how="left")
train_feature_res_df['label'] = train_feature_res_df['label'].fillna(0)

print("true lable number is:", sum(train_feature_res_df.loc[:,'label'].dropna()))
  


feature_columns = list(train_feature_res_df.columns)
feature_columns.remove('label')
feature_columns.remove('read_time_trace')
feature_columns.remove('write_time_trace')
feature_columns.remove('item_id')
feature_columns.remove('time_step')

X_train, X_val, y_train, y_val = train_test_split(train_feature_res_df.loc[:,feature_columns], \
                                    train_feature_res_df.loc[:,'label'], test_size=0.01)   


bst = LgbmPreClass(X_train, y_train, X_val, y_val).train_pre()

# save and reload model
if semionline_flag:
    bst.save_model(ML_model_root + "/" + alg_dict['dataset_name'] + "_csize"+ str(csize) + "_s" + str(staleness_bound) + "_semionline")
else:
    bst.save_model(ML_model_root + "/" + alg_dict['dataset_name'] + "_csize"+ str(csize) + "_s" + str(staleness_bound) + "_online")
#bst = lgb.Booster(model_file=ML_model_root + "/" + alg_dict['dataset_name'])  


# print alg performance information
alg_obj.print_stats()


seq_end_time = time.time()
print('ALG Total Time: {}'.format(seq_end_time - seq_start_time))


#if semionline_flag:
#    bst = lgb.Booster(model_file=ML_model_root + "/" + alg_dict['dataset_name'] + "_csize"+ str(csize) + "_s" + str(staleness_bound) + "_semionline")
##    bst.save_model(ML_model_root + "/" + alg_dict['dataset_name'] + "_semionline")
#else:
#    bst = lgb.Booster(model_file=ML_model_root + "/" + alg_dict['dataset_name'] + "_csize"+ str(csize) + "_s" + str(staleness_bound) + "_online")
#
##feature_columns = bst.feature_name()
#test_feature_df = train_feature_res_df.iloc[0:1000,].loc[:,feature_columns]
#test_predictedY = np.around(bst.predict(test_feature_df, num_iteration=bst.best_iteration), 4)
#test_predictedY_df = pd.DataFrame(test_predictedY, index = test_feature_df.index)
#test_feature_df['predictedY'] = test_predictedY_df
#test_feature_df['predictedY1'] = test_feature_df['predictedY'].apply(lambda e: 0 if e<0.5 else 1)
#
#slice_df = pd.concat([test_feature_df['predictedY1'],train_feature_res_df.iloc[0:1000,]['label']], axis=1)
#
#sum(test_feature_df['predictedY1'])
#sum(train_feature_res_df.iloc[0:1000,]['label'])

#accuracy=accuracy_score(list(train_feature_res_df.iloc[0:1000,].loc[:,'label']), test_predictedY)









