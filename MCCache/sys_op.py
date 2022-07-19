""" TCache Workload System Operation """
from multiprocessing import Queue
import time
from utils.load_dataset import load_item_univ, load_txn_univ, load_txn_seq, load_ycsb_seq
from MCCache.cache_alg import SingleAlg
import numpy as np
import json
import pdb
import os
import psycopg2
import happybase
from pymemcache.client import base
import redis
import csv

class WorkloadManager(object):
    """Manage workload for TCache
    
    """
    def __init__(self, dataset_dir: str, sys_test=True) -> None:
        super().__init__()
        # load entire workload data from dataset_dir
        item_size_path = dataset_dir + '/item_size.pkl'
        cls_item_path = dataset_dir + '/cls_item.pkl'
        txn_item_path = dataset_dir + '/txn_item.pkl'
        id_seq_path = dataset_dir + '/id_seq.npy'
        flag_seq_path = dataset_dir + '/flag_seq.npy'
        ycsb_seq_path = dataset_dir + '/transactions.dat'
        self.item_size_dict, self.cls_item_dict = load_item_univ(item_size_path, cls_item_path)
        self.txn_item_dict = load_txn_univ(txn_item_path)
        self.txn_id_seq, self.write_flag_seq = load_txn_seq(id_seq_path, flag_seq_path)
        if sys_test:
            self.ycsb_id_2_key, self.ycsb_id_2_read, self.ycsb_id_2_write = load_ycsb_seq(ycsb_seq_path)
        self.get_workload_stats(print_stats=True)


    def get_workload_stats(self, print_stats=True) -> None:
        """ Get statistics for workload data. """
        query_num = len(self.item_size_dict)
        cls_num = len(self.cls_item_dict)
        seq_len = len(self.txn_id_seq)
        read_qry_cnt, wrt_qry_cnt = 0, 0
        wrt_txn_cnt = self.write_flag_seq.sum()
        read_txn_cnt = len(self.write_flag_seq) - wrt_txn_cnt
        item_read_time_dict, item_write_time_dict = {i:[] for i in range(query_num)}, {i:[] for i in range(query_num)}
        for time_step in range(len(self.txn_id_seq)):
            txn_vec = self.txn_item_dict[self.txn_id_seq[time_step]]
            if self.write_flag_seq[time_step]:
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
        total_item_size = sum(self.item_size_dict.values())
        self.workload_stats = {'query_num': query_num, 'cls_num': cls_num, 
            'total_size': total_item_size, 'seq_len': seq_len,
            'read_txn_cnt': read_txn_cnt, 'write_txn_cnt': wrt_txn_cnt, 
            'read_qry_cnt': read_qry_cnt, 'write_qry_cnt': wrt_qry_cnt, 
            'unique_read_qry_cnt': dist_read_qry_cnt, 'unique_write_qry_cnt': dist_wrt_qry_cnt}
        
        workload_stats_columns = ['query_num', 'cls_num', 'total_size', 'seq_len',
               'read_txn_cnt', 'write_txn_cnt', 'read_qry_cnt', 'write_qry_cnt', 'unique_read_qry_cnt', 'unique_write_qry_cnt']
        dict_data = [
        {'query_num': query_num, 'cls_num': cls_num, 
            'total_size': total_item_size, 'seq_len': seq_len,
            'read_txn_cnt': read_txn_cnt, 'write_txn_cnt': wrt_txn_cnt, 
            'read_qry_cnt': read_qry_cnt, 'write_qry_cnt': wrt_qry_cnt, 
            'unique_read_qry_cnt': dist_read_qry_cnt, 'unique_write_qry_cnt': dist_wrt_qry_cnt}]
        csv_file = "data/res/workload_stats.csv"
        try:
            with open(csv_file, 'a', newline='') as p: # a means append
                writer = csv.DictWriter(p, fieldnames=workload_stats_columns)
                if os.stat(csv_file).st_size == 0: # if csv_file is empty, then add the header
                    writer.writeheader()
                for data in dict_data:
                    writer.writerow(data)
        except IOError:
            print("I/O error")  
        
        if print_stats:
            print(self.workload_stats)
            # json.dumps(self.workload_stats, indent=4)


class WorkloadTest(WorkloadManager):
    """ Create input for algorithm, operate Cache and DB accordingly. """
    def __init__(self, dataset_dir: str, sys_test=True) -> None:
        super().__init__(dataset_dir)
        self.sys_test = sys_test


    def batch_test(self, queue: Queue, csize: float, cache_scheme: str, alg_name: str, batch_size: int, opt_len: int, fetch_strategy:str, staleness_bound:int, ob_acc:float) -> dict:
        """ Test transaction sequence in batches. """
        assert self.workload_stats['seq_len'] % batch_size == 0   # transaction sequence consists of full batches by default
        cache_size =  int(csize * self.workload_stats['total_size'])
        alg_obj = SingleAlg(alg_name, fetch_strategy)
        # using opt_len as findOB truncate length for optimization
        alg_obj.workload_init(cache_size, cache_scheme, item_num=self.workload_stats['query_num'], findOB_trunc=opt_len, staleness_bound=staleness_bound, ob_acc=ob_acc)
        batch_num = int(self.workload_stats['seq_len'] / batch_size)
        seq_start_time = time.time()
        for i in range(batch_num):
            batch_start, batch_end = i * batch_size, (i + 1) * batch_size
            # get item read and write time for current batch
            alg_obj.init_read_write_time(txn_id_seq=self.txn_id_seq[batch_start:batch_end], 
                write_flag_seq=self.write_flag_seq[batch_start:batch_end], 
                txn_item_dict=self.txn_item_dict, batch_start=batch_start)
            for time_step in range(batch_start, batch_end):                
                op_ret_dict = alg_obj.batch_step_process(time_step, batch_start, batch_end, self.txn_id_seq[batch_start:batch_end], 
                    self.write_flag_seq[batch_start:batch_end], self.item_size_dict, self.txn_item_dict, self.cls_item_dict)
                if self.sys_test:
                    tmp_op_dict = {}
                    if 'write_items' in op_ret_dict:
                        query_list = [self.ycsb_id_2_write[item_id] for item_id in np.where(op_ret_dict['write_items'] == 1)[0]]
                        tmp_op_dict['db_write'] = query_list
                        tmp_op_dict['db_write_key'] = [self.ycsb_id_2_key[item_id] for item_id in np.where(op_ret_dict['write_items'] == 1)[0]]
                        # self.db_write_txn(query_list)
                    if 'acc_update' in op_ret_dict:
                        item_to_load = [item_id for item_id in np.where(op_ret_dict['acc_update'] == 1)[0]]
                        query_list = [self.ycsb_id_2_read[item_id] for item_id in item_to_load]
                        tmp_op_dict['acc_update'] = query_list
                        tmp_op_dict['acc_update_key'] = [self.ycsb_id_2_key[item_id] for item_id in item_to_load]
                    if 'evict_from_cache' in op_ret_dict:
                        tmp_op_dict['cache_evict'] = [self.ycsb_id_2_key[item_id] for item_id in np.where(op_ret_dict['evict_from_cache'] == 1)[0]]
                    if 'read_on_miss' in op_ret_dict:
                        item_to_load = [item_id for item_id in np.where(op_ret_dict['read_on_miss'] == 1)[0]]
                        query_list = [self.ycsb_id_2_read[item_id] for item_id in item_to_load]
                        tmp_op_dict['read_on_miss'] = query_list
                        tmp_op_dict['read_on_miss_key'] = [self.ycsb_id_2_key[item_id] for item_id in item_to_load]
                        # self.load_to_cache(op_ret_dict['read_on_miss'])
                    if 'read_on_abort' in op_ret_dict:
                        item_to_load = [item_id for item_id in np.where(op_ret_dict['read_on_abort'] == 1)[0]]
                        query_list = [self.ycsb_id_2_read[item_id] for item_id in item_to_load]
                        tmp_op_dict['read_on_abort'] = query_list
                        tmp_op_dict['read_on_abort_key'] = [self.ycsb_id_2_key[item_id] for item_id in item_to_load]
                        # self.load_to_cache(op_ret_dict['read_on_abort'])
                    if 'read_from_cache' in op_ret_dict:
                        tmp_op_dict['cache_read'] = [self.ycsb_id_2_key[item_id] for item_id in np.where(op_ret_dict['read_from_cache'] == 1)[0]]
                        # txn_answ = self.cache_client.get_many([self.ycsb_id_2_key[item_id] for item_id in np.where(op_ret_dict['read_from_cache'] == 1)[0]])
                    queue.put(tmp_op_dict)
        queue.put(None)
        seq_end_time = time.time()
        #
        alg_obj.print_stats()
        
        res_columns = ['alg_name', 'cache_scheme', 'cost', 'whole_cost',
               'ob_cnt', 'evict_cnt','cch_cnt']
        dict_data = [
        {'alg_name': alg_obj.alg_name, 'cache_scheme': alg_obj.cache_scheme, 'cost': alg_obj.cost, 'whole_cost': alg_obj.whole_cost,
               'ob_cnt': alg_obj.ob_cnt, 'evict_cnt': alg_obj.evict_cnt, 'cch_cnt': alg_obj.cch_cnt}]
        csv_file = "data/res/res.csv"
        try:
            with open(csv_file, 'a', newline='') as p: # a means append
                writer = csv.DictWriter(p, fieldnames=res_columns)
                if os.stat(csv_file).st_size == 0: # if csv_file is empty, then add the header
                    writer.writeheader()
                for data in dict_data:
                    writer.writerow(data)
        except IOError:
            print("I/O error")  
        
        
        print('ALG Total Time: {}'.format(seq_end_time - seq_start_time))



class SystemOperator:
    """Operate DB and cache based on operations in queue."""
    def __init__(self, db_conn_param: dict, cache_conn_param: dict) -> None:
        self.db_sys = db_conn_param["db"]
        if self.db_sys == "postgresql":
            self.db_conn = psycopg2.connect(**db_conn_param)
            self.db_conn.set_session(autocommit=False)
            self.db_cursor = self.db_conn.cursor()
        else:
            assert self.db_sys == "hbase"
            # self.db_conn = happybase.Connection(host=db_conn_param["host"], port=db_conn_param["port"])
            self.db_conn = happybase.Connection(host=db_conn_param["host"], port=9090)
            self.table = self.db_conn.table('usertable')
        self.cache_sys = cache_conn_param["cache"]
        if self.cache_sys == "memcached":
            # int((cache_conn_param['port'])) is important! Or raise error:TypeError: an integer is required (got type str)
            # https://stackoverflow.com/questions/28199521/python-typeerror-an-integer-is-required-while-working-with-sockets
            self.cache_client = base.Client((cache_conn_param['host'], int((cache_conn_param['port']))))
            self.cache_client.flush_all()
        else:
             assert self.cache_sys == "redis"
             self.cache_client = redis.Redis(host=cache_conn_param['host'], port=cache_conn_param['port'], db=0)
             self.cache_client.flushall()
    

    def operation_exe(self, op_queue: Queue):
        """Get operations from queue, execute them."""
        counter = 0
        db_read_time, db_write_time, cache_read_time, cache_write_time = 0, 0, 0, 0
        cache_counter = 0
        cache_list = []
        while True:
            op_dict = op_queue.get()
            if op_dict is None:
                print('counter: {}'.format(counter))
                sys_end_time = time.time()
                if sys_start_time is not None:
                    print('System Execution Time: {}'.format(sys_end_time - sys_start_time))
                print('db_read_time: {}, db_write_time: {}'.format(db_read_time, db_write_time))
                print('cache_read_time: {}, cache_write_time: {}'.format(cache_read_time, cache_write_time))
                break
            else:
                if counter == 0:
                    sys_start_time = time.time()
                if 'db_write' in op_dict:
                    db_write_time += self.db_write_txn(op_dict['db_write'], op_dict['db_write_key'])
                if 'acc_update' in op_dict:
                    db_read_time += self.load_to_cache(op_dict['acc_update'], op_dict['acc_update_key'])
                if 'cache_evict' in op_dict:
                    tmp_start_time = time.time()
                    if self.cache_sys == "memcached":
                        self.cache_client.delete_many(op_dict['cache_evict'])
                    else:
                        assert self.cache_sys == "redis"
                        if len(op_dict['cache_evict'])>0:
                            self.cache_client.delete(*op_dict['cache_evict'])
                    cache_write_time += time.time() - tmp_start_time
                if 'read_on_miss' in op_dict:
                    db_read_time += self.load_to_cache(op_dict['read_on_miss'], op_dict['read_on_miss_key'])
                if 'read_on_abort' in op_dict:
                    db_read_time += self.load_to_cache(op_dict['read_on_abort'], op_dict['read_on_abort_key'])
                if 'cache_read' in op_dict:
                    cache_counter += 1
                    cache_list.append(op_dict['cache_read'][0])
                    # sumbit in a batch way
                    if (counter%10==0):
                        tmp_start_time = time.time()
                        if self.cache_sys == "memcached":
                            self.cache_client.get_many(cache_list) 
                        else:
                            assert self.cache_sys == "redis"
                            self.cache_client.mget(cache_list)
                        cache_read_time += time.time() - tmp_start_time
                counter += 1


    def load_to_cache(self, query_list: list, key_str_list: list):
        """Execute read transaction at database and load results to cache."""
        tmp_load_start_time = time.time()
        if self.db_sys == "postgresql":
            try:
                txn_res_list = []
                for query_str in query_list:
                    self.db_cursor.execute(query_str)
                    txn_res_list.extend(self.db_cursor.fetchall())
                self.db_conn.commit()
                for query_result in txn_res_list:
                    self.cache_client.set(query_result[0], ''.join(query_result[1:]))
            except psycopg2.DatabaseError as e:
                print('PostgreSQL read transaction execution error')
                self.db_conn.rollback()
        else:   # HBase read transaction
            try:
                txn_res_list = []
                for (key_str, qual_bytes_list) in zip(key_str_list, query_list):
                    row = self.table.row(key_str.encode('utf-8'), columns=qual_bytes_list)
                    txn_res_list.append((key_str, b''.join(row.values()).decode('utf-8')))
                    if self.cache_sys == "memcached":
                        self.cache_client.set_many(dict(txn_res_list))
                    else:
                        assert self.cache_sys == "redis"
                        self.cache_client.mset(dict(txn_res_list))
            except ValueError:
                print('HBase value error.')
                pass
        tmp_load_end_time = time.time()
        return tmp_load_end_time - tmp_load_start_time
    

    def db_write_txn(self, query_list: list, key_str_list: list):
        """Execute write transaction at database."""
        tmp_write_start_time = time.time()
        if self.db_sys == "postgresql": # postgresql write transaction
            try:
                for query_str in query_list:
                    self.db_cursor.execute(query_str)
                self.db_conn.commit()
            except psycopg2.DatabaseError as e:
                print('PostgreSQL write transaction execution error')
                self.db_conn.rollback()
        else:   # HBase write transaction
            try:
                with self.table.batch(transaction=True) as bat:
                    for (key_str, qual_val_dict) in zip(key_str_list, query_list):
                        bat.put(key_str.encode('utf-8'), qual_val_dict)
            except ValueError:
                print('HBase value error.')
                pass
        tmp_write_end_time = time.time()
        return tmp_write_end_time - tmp_write_start_time


    def cleanup(self):
        """Close connection after testing."""
        self.db_conn.close()
        if self.cache_sys == "memcached":
             self.cache_client.flush_all()
             self.cache_client.close()
        else:
             assert self.cache_sys == "redis"
             self.cache_client.flushall()
             self.cache_client.connection_pool.disconnect()
       