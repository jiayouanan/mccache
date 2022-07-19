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

import os


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
    # item_num = config['item_num']
    item_num = config['hot_item_num'] + config['cold_item_num']
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
        item_size_arr = rng.integers(low=item_min_size, high=item_max_size, size=item_num)
        np.random.shuffle(item_size_arr)
        for j in range(item_num):
            item_size_dict[j] = item_size_arr[j]
    else:
        raise ValueError('Undefined item distribution.')
    print('Item Size Dict: \n {}'.format(item_size_dict))
    # generate cls_item
    cls_item_fp = open(size_res_path, 'wb')
    pickle.dump(item_size_dict, cls_item_fp)
    cls_item_fp.close()
    # compute cls_item_dict based on item_size_dict
    cls_item_dict = {}
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

 
def genr_txn_univ(config: dict, res_path='data/txn_item.pkl'):
    """Generate universe of transactions (set of queries).

    Generate and save <Transaction Query Table>. Transaction total number and 
    the number of queries in each transaction are specified in params. 
    <Transaction Query Table> map transaction id to its query content.

    Args:
        config: dict, specified params.
        res_path: str, file path to save <Transaction Query Table>.

    Returns:
        A dict mapping transaction id to boolean vector showing which 
        queries are in that transaction.
    
    Raises:
        ValueError: Undefined transaction (set) distribution.
    """
    item_num = config_dict['item_num']
    txn_num = config['set_num']  # transaction number
    txn_size_min = config['set_size']['min']  # transaction minimum query number
    txn_size_max = config['set_size']['max']  # transaction maximum query number
    txn_item_dict = {}  # <Transaction Query Table>
    if config['set_distr'] == "plain_random":
        print('Generating transaction universe in plain_random manners.')
        # decide unique transaction number for each transaction size by random
        txn_size_num = txn_size_max - txn_size_min + 1
        while True:
            rng = default_rng(216)
            samples = rng.choice(txn_num, txn_size_num - 1)
            if 0 not in samples and txn_num not in samples:
                break
        samples.sort()
        group_cnt_thresh = np.concatenate((np.array([0]), samples, np.array([txn_num])))
        # list of transaction group count, each group have same number of queries in each transaction
        txn_group_cnt = [group_cnt_thresh[i+1] - group_cnt_thresh[i] for i in range(txn_size_num)]
        txn_id_list = [j for j in range(txn_num)]
        random.shuffle(txn_id_list)
        unique_txn_cnt = 0
        rng = default_rng(216)  # numpy random generator
        for txn_group_id in range(txn_size_num):
            txn_cnt_2_genr = txn_group_cnt[txn_group_id]    # num of txns to generate for this group
            tmp_txn_size = txn_size_min + txn_group_id      # num of queries each txn contains
            group_txn_set = set()   # store frozenset representing unique txns
            tmp_unique_txn_cnt = 0     # temperary counter for this group's unique txns
            while True:
                tmp_set = frozenset(rng.choice(item_num, tmp_txn_size))
                if tmp_set not in group_txn_set:
                    group_txn_set.add(tmp_set)
                    tmp_unique_txn_cnt += 1
                if tmp_unique_txn_cnt == txn_cnt_2_genr:
                    break
            for tmp_set in group_txn_set:
                txn_vec = np.zeros(item_num, dtype=bool)
                for item_id in tmp_set:
                    txn_vec[item_id] = 1
                txn_id = txn_id_list[unique_txn_cnt]
                txn_item_dict[txn_id] = txn_vec
                unique_txn_cnt += 1
        assert unique_txn_cnt == txn_num    # $txn_num$ unique txns should be generated
    else:
        raise ValueError('Undefined transaction (set) distribution.')
    txn_item_fp = open(res_path, 'wb')
    pickle.dump(txn_item_dict, txn_item_fp)
    txn_item_fp.close()
    return txn_item_dict


def genr_txn_seq(config: dict, id_seq_path='data/id_seq.npy', flag_seq_path='data/flag_seq.npy'):
    """Generate read write transaction sequence.

    Generate transaction id sequence (int array) and write flag sequence 
    (boolean array). Write flag is 1 (True) for write transactions.

    Args:
        config: dict, specify params.
        id_seq_path: str, file path to save id sequence.
        flag_seq_path: str, file path to save write flat sequence.
    
    Returns:
        int array of id sequence
        boolean array of write flag sequence
    """
    txn_num = config['set_num']     # num of unique transactions
    seq_len = config['seq_len']  # read & write transaction sequence length
    write_freq = config['write_freq']   # expected write transaction frequence
    if config['seq_distr'] == "plain_random":
        rng = default_rng(216)
        id_seq = rng.integers(low=0, high=txn_num, size=seq_len)
        flag_arr = rng.random(seq_len)
        write_flag_seq = flag_arr < write_freq  # Boolean arr, True for write transaction
    else:
        raise ValueError('Undefined transaction sequence distribution.')
    np.save(id_seq_path, id_seq)
    np.save(flag_seq_path, write_flag_seq)
    return id_seq, write_flag_seq


def genr_qry_seq(config: dict, txn_uni_path='data/txn_item.pkl', txn_id_seq_path='data/id_seq.npy', flag_seq_path='data/flag_seq.npy'):
    """Generate query sequence and group them into transactions.

    Generate query sequence or transaction one at a time. Create transaction universe, 
    transaction id sequence and write flag sequence along the process.

    Args: 
        config: dict, specify params.
        txn_uni_path: str, file path to save transaction universe (txn-id:qry-vec).
        txn_id_seq_path: str, file path to save transaciton id sequence.
        flag_seq_path: str, file path to save write-read flag sequence.
    """
    # TODO: define other complicated transaction patterns
    """ generate write flag sequence """
    seq_len = config['seq_len']  # transaction sequence length (read and write transactions)
    write_freq = config['write_freq']  # expected write transaction frequency
    rng = default_rng(418)
    flag_arr = rng.random(seq_len)
    write_flag_seq = flag_arr < write_freq  # boolean array, True for write transaction
    np.save(flag_seq_path, write_flag_seq)  # save to numpy file
    """ generate read write transaction size array """
    read_txn_size_min, read_txn_size_max = config['read_txn_size']['min'], config['read_txn_size']['max']
    write_txn_size_min, write_txn_size_max = config['write_txn_size']['min'], config['write_txn_size']['max']
    read_txn_cnt, write_txn_cnt = seq_len - write_flag_seq.sum(), write_flag_seq.sum()
    read_txn_size_arr = rng.integers(low = read_txn_size_min, high = read_txn_size_max + 1, size = read_txn_cnt)
    write_txn_size_arr = rng.integers(low = write_txn_size_min, high = write_txn_size_max + 1, size = write_txn_cnt)
    """ generate read write transaction one by one """
    hot_item_num, cold_item_num = config['hot_item_num'], config['cold_item_num']
    item_num = hot_item_num + cold_item_num
    unique_txn_cnt = 0
    read_txn_idx, write_txn_idx = 0, 0
    txn_item_dict = {}  # <Transaction Query Table>
    vec_to_txn_id = {}  # <Query vector to Transaction id Table>, numpy array to bytes as key
    txn_id_seq = np.zeros(seq_len, dtype = int)
    # take hot-cold split into consideration
    hot_pct = config['hot_cold_freq_split'] / 100
    hot_cold_arr = rng.random(seq_len)  # decide to round up or down hot query number in each transaction
    for write_flag in write_flag_seq:
        if write_flag:
            write_txn_size = write_txn_size_arr[write_txn_idx]
            txn_hot_num = int(hot_pct * write_txn_size) + 1 if int(hot_pct * write_txn_size + hot_cold_arr[write_txn_idx]) > int(hot_pct * write_txn_size) else int(hot_pct * write_txn_size)
            txn_cold_num = write_txn_size - txn_hot_num
            chosen_hot_qry, chosen_cold_qry = rng.choice(hot_item_num, size=txn_hot_num), rng.choice(cold_item_num, size=txn_cold_num)
            chosen_cold_qry += hot_item_num
            item_vec = np.zeros(item_num, dtype=bool)
            for hot_qry_idx in chosen_hot_qry:
                item_vec[hot_qry_idx] = 1
            for cold_qry_idx in chosen_cold_qry:
                item_vec[cold_qry_idx] = 1
            if item_vec.tobytes() in vec_to_txn_id:
                txn_id = vec_to_txn_id[item_vec.tobytes()]
            else:
                txn_id = unique_txn_cnt
                txn_item_dict[txn_id] = item_vec
                vec_to_txn_id[item_vec.tobytes()] = txn_id
                unique_txn_cnt += 1
            txn_id_seq[read_txn_idx + write_txn_idx] = txn_id
            write_txn_idx += 1
        else:
            read_txn_size = read_txn_size_arr[read_txn_idx]
            txn_hot_num = int(hot_pct * read_txn_size) + 1 if int(hot_pct * read_txn_size + hot_cold_arr[read_txn_idx]) > int(hot_pct * read_txn_size) else int(hot_pct * read_txn_size)
            txn_cold_num = read_txn_size - txn_hot_num
            chosen_hot_qry, chosen_cold_qry = rng.choice(hot_item_num, size=txn_hot_num), rng.choice(cold_item_num, size=txn_cold_num)
            chosen_cold_qry += hot_item_num
            item_vec = np.zeros(item_num, dtype=bool)
            for hot_qry_idx in chosen_hot_qry:
                item_vec[hot_qry_idx] = 1
            for cold_qry_idx in chosen_cold_qry:
                item_vec[cold_qry_idx] = 1
            if item_vec.tobytes() in vec_to_txn_id:
                txn_id = vec_to_txn_id[item_vec.tobytes()]
            else:
                txn_id = unique_txn_cnt
                txn_item_dict[txn_id] = item_vec
                vec_to_txn_id[item_vec.tobytes()] = txn_id
                unique_txn_cnt += 1
            txn_id_seq[read_txn_idx + write_txn_idx] = txn_id
            read_txn_idx += 1
    """ save transaction id sequence and write flag sequence to file """
    print('unique_txn_cnt: {}'.format(unique_txn_cnt))
    np.save(flag_seq_path, write_flag_seq)
    np.save(txn_id_seq_path, txn_id_seq)
    txn_item_fp = open(txn_uni_path, 'wb+')
    pickle.dump(txn_item_dict, txn_item_fp)
    txn_item_fp.close()
    # debug
    a = np.ones(item_num, dtype=bool)
    for i in range(unique_txn_cnt):
        a = np.logical_or(a, txn_item_dict[i])
    print('actual query number: {}'.format(a.sum()))


if __name__ == '__main__':
    path = abspath(dirname(dirname(__file__)))+'/config/TCBench_mod.yaml'
    config_file = open(path, 'r')
    config_dict = yaml.load(config_file, Loader=yaml.FullLoader)

    hot_item_num, cold_item_num = config_dict['hot_item_num'], config_dict['cold_item_num']
    hot_cold_split = config_dict['hot_cold_freq_split']
    item_max_size = config_dict['item_max_size']
    read_txn_size_min, read_txn_size_max = config_dict['read_txn_size']['min'], config_dict['read_txn_size']['max']
    write_txn_size_min, write_txn_size_max = config_dict['write_txn_size']['min'], config_dict['write_txn_size']['max']
    write_freq = config_dict['write_freq']

    workload_dir = abspath(dirname(dirname(__file__))) + '/data/TCBench_mod/hot{}_cold{}_split{}_itemmaxsize{}_readmin{}_readmax{}_writemin{}_writemax{}_wrtpct{}'.format(hot_item_num, cold_item_num, hot_cold_split, item_max_size, read_txn_size_min, read_txn_size_max, write_txn_size_min, write_txn_size_max, write_freq)

    Path(workload_dir).mkdir(parents=True, exist_ok=True)
    
    size_res_path = workload_dir + '/item_size.pkl'
    cls_res_path = workload_dir + '/cls_item.pkl'
    res_path = workload_dir + '/txn_item.pkl'
    id_seq_path = workload_dir + '/id_seq.npy'
    flag_seq_path = workload_dir + '/flag_seq.npy'

    # size_res_path = abspath(dirname(dirname(__file__))) + '/data/item_size.pkl'
    # cls_res_path = abspath(dirname(dirname(__file__))) + '/data/cls_item.pkl'
    # res_path = abspath(dirname(dirname(__file__))) + '/data/txn_item.pkl'
    # id_seq_path = abspath(dirname(dirname(__file__))) + '/data/id_seq.npy'
    # flag_seq_path = abspath(dirname(dirname(__file__))) + '/data/flag_seq.npy'
    
    # genr_item_univ(config_dict, size_res_path, cls_res_path)
    # genr_txn_univ(config_dict, res_path)
    # genr_txn_seq(config_dict, id_seq_path, flag_seq_path)

    genr_item_univ(config_dict, size_res_path, cls_res_path)
    genr_qry_seq(config_dict, res_path, id_seq_path, flag_seq_path)
