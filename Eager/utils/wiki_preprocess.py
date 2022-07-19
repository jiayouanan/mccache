# %%
import numpy as np
from numpy.random import default_rng
import multiprocessing as mp
import pdb
from itertools import islice
import time
import math
from pathlib import Path
import pickle


def load_wiki_one_batch(fp, batch_size):
    """Load wikipedia trace in batch, return batch_num*batch_size records (lines) as list

    """
    bat_lines = []
    for lines in iter(lambda: list(islice(fp, batch_size)), []):
        bat_lines.extend([x.decode('utf-8').strip().split() for x in lines])
        break
    return bat_lines


def wiki_tr_to_npy():
    trace_fp = open('data/wiki2018.tr', 'rb')
    batch_num = 100
    batch_size = 1000000
    for r in range(28):
        start_time = time.time()
        print('round: {}/28'.format(r + 1))
        wiki_npy_arr = np.empty((batch_num * batch_size, 4), dtype=int)
        wiki_npy_file = 'data/wiki_trace_BN{}_BS{}_RND{}.npy'.format(batch_num, batch_size, r)
        for i in range(batch_num):
            wiki_npy_arr[i * batch_size:(i + 1) * batch_size] = np.array(load_wiki_one_batch(trace_fp, batch_size), dtype=int)
        np.save(wiki_npy_file, wiki_npy_arr)
        print('round trace npy save time: {}'.format(time.time() - start_time))


def mp_get_wiki_sample(batch_trace_arr, threshold, sample_num, batch_id):
    """Workhorse for mp_wiki_trace_filter. Sample over one batch/partition of wiki_trace_arr.

    """
    item_id_arr, item_cnt_arr = np.unique(batch_trace_arr[:, 1], return_counts=True)   # count item id appear time
    sel_items = item_id_arr[item_cnt_arr >= threshold]  # selected item id to preserve
    batch_trace_arr = batch_trace_arr[np.isin(batch_trace_arr[:, 1], sel_items)]
    rng = default_rng(311)
    samples = rng.choice(batch_trace_arr.shape[0], int(sample_num))
    samples.sort()
    return batch_trace_arr[samples], batch_id


def mp_wiki_trace_filter(wiki_trace_arr, threshold, sample_num):
    """Multi-process wiki trace filter method.
    
    Args:
        wiki_trace_arr: original (one partition) of wikipedia trace.
        threshold: items in wiki_trace_arr appear over #threshold times are selected.
        sample_num: number of traces selected from wiki_trace_arr.
    """
    pool = mp.Pool(mp.cpu_count())
    batch_num = 5
    batch_size = int(wiki_trace_arr.shape[0] / batch_num)
    filter_result = pool.starmap(mp_get_wiki_sample, [[wiki_trace_arr[i * batch_size : (i + 1) * batch_size], int(threshold / batch_num), int(sample_num / batch_num), i] for i in range(batch_num)])
    sample_vec = np.zeros(shape = (sample_num, 6), dtype = int)
    sample_batch_size = int(sample_num / batch_num)
    for filter_result_batch in filter_result:
        batch_id = filter_result_batch[1]
        sample_vec[batch_id * sample_batch_size : (batch_id + 1) * sample_batch_size, 0 : 4] = filter_result_batch[0]
    return sample_vec


def get_item_universe(sample_vec):
    """Get item_size_dict, cls_item_dict from filtered wiki trace samples.
    
    """
    unique_item_arr, unique_item_cnt = np.unique(sample_vec[:, 1], return_counts=True)
    item_num = unique_item_arr.shape[0]
    unique_size_arr = np.unique(sample_vec[:, 2])
    min_item_size, max_item_size = unique_size_arr.min(), unique_size_arr.max()
    cls_num = math.ceil(math.log2(max_item_size / min_item_size))
    cls_item_dict = {}
    for i in range(cls_num):
        cls_item_dict[i] = np.zeros(item_num, dtype=bool)
    item_size_dict = {}
    for item_id in range(item_num):
        item_size = sample_vec[np.where(sample_vec[:, 1] == unique_item_arr[item_id])][0, 2]
        item_size_dict[item_id] = item_size
        item_cls = math.floor(math.log2(item_size / min_item_size))
        cls_item_dict[item_cls][item_id] = 1
    return item_num, cls_num, item_size_dict, cls_item_dict, unique_item_arr, unique_item_cnt


def get_wiki_txn(sample_vec, txn_low, txn_high):
    """Group wiki trace sample vector into transactions. Write flag not included.

    """
    # generate #trace_len random integers, as sufficient transaction groups for wiki trace
    rng = default_rng(313)
    trace_len = sample_vec.shape[0]
    group_seq = rng.integers(low=txn_low, high=txn_high+1, size=trace_len)
    group_sum = 0
    for group_idx in range(trace_len):
        group_size = group_seq[group_idx]
        # assign transaction time step to wiki trace sample records
        sample_vec[group_sum : group_sum + group_size, 4] = group_idx
        group_sum += group_size
        if group_sum >= trace_len:
            break

new_sample = False
start_time = time.time()
rnd = 21
wiki_trace_len = 100 * 1000000
threshhold = int(wiki_trace_len / 5000)
sample_num = int(wiki_trace_len / 500)
sample_npy_file = 'data/wiki2018_THR{}_SN{}_RND{}.npy'.format(threshhold, sample_num, rnd)

if new_sample:
    wiki_trace_npy_file = 'data/wiki_trace_BN100_BS1000000_RND{}.npy'.format(rnd)
    wiki_trace_arr = np.load(wiki_trace_npy_file, allow_pickle=True)
    sample_vec = mp_wiki_trace_filter(wiki_trace_arr, threshhold, sample_num)
    np.save(sample_npy_file, sample_vec, allow_pickle=True)
    print('sample_time: {}'.format(time.time() - start_time))

sample_vec = np.load(sample_npy_file, allow_pickle=True)

# check original sample_vec item size distribution
item_num, cls_num, item_size_dict, cls_item_dict, unique_item_arr, unique_item_cnt = get_item_universe(sample_vec)
cls_cnt = [cls_item_dict[i].sum() for i in range(cls_num)]
print('original cls--item count: {}'.format(cls_cnt))

cls_appear_dict = {}
for cls_id in cls_item_dict:
    cls_item_arr = np.where(cls_item_dict[cls_id] == 1)[0]
    cls_cnt = 0
    for i in range(len(cls_item_arr)):
        item_id = cls_item_arr[i]
        item_cnt = unique_item_cnt[item_id]
        cls_cnt += item_cnt
    cls_appear_dict[cls_id] = cls_cnt

print('cls--appear count: {}'.format(cls_appear_dict))

# %%
chosen_cls_list = [i for i in range(0, 8)]
cls_limit = len(chosen_cls_list)
sample_min_size = sample_vec[:, 2].min()
min_item_size = sample_min_size * pow(2, chosen_cls_list[0])
max_item_size = sample_min_size * pow(2, (chosen_cls_list[-1] + 1))
sample_idx = np.logical_and(sample_vec[:, 2] >= min_item_size, sample_vec[:, 2] < max_item_size)
sample_vec = sample_vec[sample_idx]
cls_cnt = [cls_item_dict[i].sum() for i in chosen_cls_list]
print('limited cls--item count: {}'.format(cls_cnt))


# generate transaction with duplication, flexible size
txn_len = 10000
txn_low = 8
txn_high = 12
get_wiki_txn(sample_vec, txn_low, txn_high)

# no write flag here, assigned upon testing
write_freq = 0
write_flag_seq = np.zeros(txn_len, dtype=bool)

# eliminate duplicate items in each transaction
unique_txn_cnt = 0  # unique transaction count, used as current transaction id
txn_id_seq = np.zeros(txn_len, dtype=int)
txn_item_dict = {}
item2txn = {}


tmp_i, i = 0, 0
while i < txn_len:
    tmp_txn_arr = sample_vec[sample_vec[:, 4] == tmp_i]
    txn_item_dict[unique_txn_cnt] = np.zeros(item_num, dtype=bool)
    item_id_set = frozenset([np.where(unique_item_arr == wiki_id)[0][0] for wiki_id in tmp_txn_arr[:, 1]])
    if len(item_id_set) < txn_low:
        tmp_i += 1
        continue
    if item_id_set not in item2txn:
        txn_item_dict[unique_txn_cnt][list(item_id_set)] = 1
        item2txn[item_id_set] = unique_txn_cnt
        unique_txn_cnt += 1
    tmp_txn_id = item2txn[item_id_set]
    txn_id_seq[i] = tmp_txn_id
    i += 1
    tmp_i += 1
    if i % 1000 == 0:
        print('{}/{}'.format(i, txn_len))

print('unique txn count: {}'.format(unique_txn_cnt))
txn_cnt = [txn_item_dict[i].sum() for i in range(unique_txn_cnt)]
print('txn--item count min: {}, max: {}, mean: {}'.format(min(txn_cnt), max(txn_cnt), sum(txn_cnt)/txn_len))

# %%
# save transaction workload
workload_dir = 'data/WikiTrace/'
Path(workload_dir).mkdir(parents=True, exist_ok=True)  # make workload dir if not exist
item_size_file = workload_dir + '/item_size.pkl'
cls_item_file = workload_dir + '/cls_item.pkl'
txn_item_file = workload_dir + '/txn_item.pkl'
txn_id_file = workload_dir + '/id_seq.npy'
write_flag_file = workload_dir + '/flag_seq.npy'
item_size_fp = open(item_size_file, 'wb')
pickle.dump(item_size_dict, item_size_fp)
item_size_fp.close()
cls_item_fp = open(cls_item_file, 'wb')
pickle.dump(cls_item_dict, cls_item_fp)
cls_item_fp.close()
txn_item_fp = open(txn_item_file, 'wb')
pickle.dump(txn_item_dict, txn_item_fp)
txn_item_fp.close()
np.save(txn_id_file, txn_id_seq)
np.save(write_flag_file, write_flag_seq)
print('workload generation time: {}'.format(time.time() - start_time))