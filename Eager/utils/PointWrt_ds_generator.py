'''
    Generate transaction sequence and other necessary input information
'''
import yaml
import argparse
import numpy as np
from numpy.random import default_rng
import random
import math
import pickle
import pdb
import scipy.stats as stats
from os.path import dirname
from os.path import abspath
from pathlib import Path


def genr_item_univ(config: dict, size_res_path='data/item_size.pkl', cls_res_path='data/cls_item.pkl'):
    """Generate universe of items (queries).

    Generate and save result as <Item Size Table>. Item number and size range 
    specified by params in config. Compute and save <Class Item Table> based on 
    <Item Size Table> result.
    
    Args:
        config: dict, params parsed from input_params.yaml file
        size_res_path: str, file path to save <Item Size Table> dict
        cls_res_path: str, file path to save <Class Item Table> dict
    
    Returns:
        a dict mapping item id to item size,
        another dict mapping class id to class vector (binary/boolean) 
        showing which items are in each class.
    
    Raises:
        ValueError: Undefined item distribution.
    """
    item_size_dict = {}
    # cls_num = config['item_cls_num']
    item_num = config['item_num']
    item_min_size = config['item_min_size']  # s0, minimum item size
    item_max_size = config['item_max_size']  # s1, maximum item size
    
    if config['item_distr'] == 'cls_norm':
        # assert item_num % cls_num == 0  # each class contains same number of items
        print('Generating item universe in cls_norm distribution.')
        mu = (item_min_size + item_max_size) / 2 # adopt mean value of minimum and maximum size as mu
        sigma = (mu - item_min_size) / 4 # adopt 4 std, guarantee 99.99% size greater than item_min_size and smaller than item_max_size
        # random numbers satisfying normal distribution
        rng = stats.truncnorm(
        (item_min_size - mu) / sigma, (item_max_size - mu) / sigma, loc=mu, scale=sigma) 
        item_size_arr = np.around(rng.rvs(item_num), 0)
        np.random.shuffle(item_size_arr)
        for j in range(item_num):
            item_size_dict[j] = item_size_arr[j]
    elif config['item_distr'] == 'cls_random':
        rng = default_rng(216)      # set random seed
        item_size_arr = rng.integers(low=item_min_size, high=item_max_size + 1, size=item_num)
        np.random.shuffle(item_size_arr)
        for j in range(item_num):
            item_size_dict[j] = item_size_arr[j]
    elif config['item_distr'] == 'uni_size':
        assert item_min_size == item_max_size
        for j in range(item_num):
            item_size_dict[j] = item_min_size
    else:
        raise ValueError('Undefined item distribution.')
    print('Item Size Dict: \n {}'.format(item_size_dict))
    # generate cls_item
    cls_item_fp = open(size_res_path, 'wb')
    pickle.dump(item_size_dict, cls_item_fp)
    cls_item_fp.close()
    # compute cls_item_dict based on item_size_dict
    cls_item_dict = {}
    if config['item_distr'] == 'uni_size':
        cls_num = 1
    else:
        cls_num = math.ceil(math.log2(item_max_size / item_min_size))
    for cls_id in range(cls_num):
        cls_item_dict[cls_id] = np.zeros(item_num, dtype=bool)
    # check each item size and update class binary vector
    for item_id in range(item_num):
        item_cls = math.floor(math.log2(item_size_dict[item_id] / item_min_size))
        cls_item_dict[item_cls][item_id] = 1
    # dump <Class Item Table> using pickle
    cls_item_fp = open(cls_res_path, 'wb')
    pickle.dump(cls_item_dict, cls_item_fp)
    cls_item_fp.close()
    return item_size_dict, cls_item_dict


def genr_txn_seq(config: dict, txn_item_path='data/txn_item.pkl', id_seq_path='data/id_seq.npy', flag_seq_path='data/flag_seq.npy'):
    """Generate transaction sequence for point-write workload.

    """
    # generate read-write flag based on write frequency
    seq_len = config['seq_len'] # transaction sequence length
    write_freq = config['write_freq']   # expected write transaction frequency
    rng = default_rng(522)
    flag_arr = rng.random(seq_len)
    write_flag_seq = flag_arr < write_freq
    np.save(flag_seq_path, write_flag_seq)  # save to numpy file
    # create read / write transactions based on recent read / write queries
    item_num = config['item_num']
    recent_read_thresh, recent_write_thresh = config['recent_read_thresh'], config['recent_write_thresh']
    read_txn_size, write_txn_size = config['read_txn_size'], config['write_txn_size']
    txn_vec_set = set() # store frozenset representing unique transactions
    unique_txn_cnt = 0  # unique transaction count, serve as transaction id during generation
    txn_item_dict = {}  # map transaction id to transaction item vector
    txn_id_seq = np.zeros(seq_len, dtype=int)   # transaction id sequence
    for i in range(seq_len):
        # generate write transaction
        if write_flag_seq[i]:
            # check if there are enought 'recent read transactions' to select write queries
            past_reads = np.where(write_flag_seq[0:i+1] == 0)[0]
            if past_reads.shape[0] >= recent_read_thresh:
                recent_reads = past_reads[-recent_read_thresh:past_reads.shape[0]]# find recent read queries
                recent_read_id = txn_id_seq[recent_reads]
                recent_read_queries = np.zeros(item_num, dtype=bool)
                for txn_id in recent_read_id:
                    recent_read_queries = np.logical_or(recent_read_queries, txn_item_dict[txn_id])
                non_recent_read_queries = np.logical_not(recent_read_queries)
                # choose 1 query from non-recent read queries, another from recent read queries
                if write_txn_size == 2:
                    recent_num, non_recent_num = 1, 1
                else:
                    # TODO: fix this 50/50 setup
                    recent_num, non_recent_num = math.ceil(write_txn_size * 0.5), math.floor(write_txn_size * 0.5)
                recent_samples = rng.choice(np.where(recent_read_queries == 1)[0], recent_num)
                non_recent_samples = rng.choice(np.where(non_recent_read_queries == 1)[0], non_recent_num)
                samples = np.concatenate((recent_samples, non_recent_samples))
                tmp_txn_vec = np.zeros(item_num, dtype=bool)
                for item_id in samples:
                    tmp_txn_vec[item_id] = 1
                tmp_item_set = frozenset(samples)
                if tmp_item_set not in txn_vec_set:
                    txn_vec_set.add(tmp_item_set)
                    tmp_txn_id = unique_txn_cnt
                    txn_id_seq[i] = tmp_txn_id
                    txn_item_dict[tmp_txn_id] = tmp_txn_vec
                    unique_txn_cnt += 1
                else:
                    for txn_id in txn_item_dict:
                        if np.equal(tmp_txn_vec, txn_item_dict[txn_id]).all():
                            txn_id_seq[i] = txn_id
                            break
            # not enough recent read transactions, choose write queries randomly
            else:
                samples = rng.choice(item_num, write_txn_size)  # choose queries by random
                tmp_txn_vec = np.zeros(item_num, dtype=bool)
                for item_id in samples:
                    tmp_txn_vec[item_id] = 1
                tmp_item_set = frozenset(samples)
                if tmp_item_set not in txn_vec_set:
                    txn_vec_set.add(tmp_item_set)
                    tmp_txn_id = unique_txn_cnt
                    txn_id_seq[i] = tmp_txn_id
                    txn_item_dict[tmp_txn_id] = tmp_txn_vec
                    unique_txn_cnt += 1
                else:
                    for txn_id in txn_item_dict:
                        if np.equal(tmp_txn_vec, txn_item_dict[txn_id]).all():
                            txn_id_seq[i] = txn_id
                            break
        # generate read transaction
        else:
            past_writes = np.where(write_flag_seq[0:i+1] == 1)[0]
            if past_writes.shape[0] >= recent_write_thresh:
                recent_writes = past_writes[-recent_write_thresh:past_writes.shape[0]]# find recent write queries
                recent_write_id = txn_id_seq[recent_writes]
                recent_write_queries = np.zeros(item_num, dtype=bool)
                for txn_id in recent_write_id:
                    recent_write_queries = np.logical_or(recent_write_queries, txn_item_dict[txn_id])
                non_recent_write_queries = np.logical_not(recent_write_queries)
                # choose 2 queries from non_recent_write, others from recent_write
                recent_num, non_recent_num = read_txn_size - 2, 2
                recent_samples = rng.choice(np.where(recent_write_queries == 1)[0], recent_num)
                non_recent_samples = rng.choice(np.where(non_recent_write_queries == 1)[0], non_recent_num)
                samples = np.concatenate((recent_samples, non_recent_samples))
                tmp_txn_vec = np.zeros(item_num, dtype=bool)
                for item_id in samples:
                    tmp_txn_vec[item_id] = 1
                tmp_item_set = frozenset(samples)
                if tmp_item_set not in txn_vec_set:
                    txn_vec_set.add(tmp_item_set)
                    tmp_txn_id = unique_txn_cnt
                    txn_id_seq[i] = tmp_txn_id
                    txn_item_dict[tmp_txn_id] = tmp_txn_vec
                    unique_txn_cnt += 1
                else:
                    for txn_id in txn_item_dict:
                        if np.equal(tmp_txn_vec, txn_item_dict[txn_id]).all():
                            txn_id_seq[i] = txn_id
                            break
            # not enough recent write transactions, choose read queries randomly
            else:
                samples = rng.choice(item_num, read_txn_size)  # choose queries by random
                tmp_txn_vec = np.zeros(item_num, dtype=bool)
                for item_id in samples:
                    tmp_txn_vec[item_id] = 1
                tmp_item_set = frozenset(samples)
                if tmp_item_set not in txn_vec_set:
                    txn_vec_set.add(tmp_item_set)
                    tmp_txn_id = unique_txn_cnt
                    txn_id_seq[i] = tmp_txn_id
                    txn_item_dict[tmp_txn_id] = tmp_txn_vec
                    unique_txn_cnt += 1
                else:
                    for txn_id in txn_item_dict:
                        if np.equal(tmp_txn_vec, txn_item_dict[txn_id]).all():
                            txn_id_seq[i] = txn_id
                            break
    # save results to file
    txn_item_fp = open(txn_item_path, 'wb')
    pickle.dump(txn_item_dict, txn_item_fp)
    txn_item_fp.close()
    np.save(id_seq_path, txn_id_seq)
    np.save(flag_seq_path, write_flag_seq)
    return txn_item_dict, txn_id_seq, write_flag_seq


if __name__ == '__main__':
    path = abspath(dirname(dirname(__file__)))+'/config/PointWrt.yaml'
    config_file = open(path, 'r')
    config_dict = yaml.load(config_file, Loader=yaml.FullLoader)

    workload_dir = abspath(dirname(dirname(__file__))) + '/data/PointWrt/' + 'QueryNum{}_Unisize_RThresh{}_WThresh{}_RSize{}_WSize{}_Wrt{}_Len{}'.format(config_dict['item_num'], config_dict['recent_read_thresh'], config_dict['recent_write_thresh'], config_dict['read_txn_size'], config_dict['write_txn_size'], config_dict['write_freq'], config_dict['seq_len'])

    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    item_size_dict, cls_item_dict = genr_item_univ(config_dict, size_res_path=workload_dir+'/item_size.pkl', cls_res_path=workload_dir+'/cls_item.pkl')
    txn_item_dict, txn_id_seq, write_flag_seq = genr_txn_seq(config_dict, txn_item_path=workload_dir+'/txn_item.pkl', id_seq_path=workload_dir+'/id_seq.npy', flag_seq_path=workload_dir+'/flag_seq.npy')
