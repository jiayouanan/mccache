# Readme for mccache

## Background
 
This repository releases the source codes and datasets for mccache paper (Caching with Monotonicity and Consistency).

Note that if you want to test the throughput results of cache policies as shown in paper, you have to correctly install and configure the Redis (or Memcached), Hbase and AWS EC2 first (see below for details). However, it is easy to get the #dbread (the number of read operations that are carried out at HBase) results without the system configurations. The results over #dbread indeed reflect the throughput results to a large extent. Intuitively, the smaller the #dbread, the more reads cache node takes. This will enable the database server to take on more transactions requested by clients, resulting in higher throughput. 


## Basic Requirements

If you want to run code without system configurations, **you still need to install the following python packages** based on python 3.6 (or higher).

```
numpy 1.20.1
happybase 1.2.0
pymemcache 3.5.0
psycopg2 2.9.1
scikit-learn 0.20.1
lightgbm 2.2.3
redis 4.3.4
```

If you use the Ubuntu system, just run below command. All the basic configurations will be successfully configured. 

```
bash init.sh
```

## Usage

After installing above packages, just run the command to obtain #dbread (cost):
```
python3 system_test.py
```
or
```
python system_test.py
```

The result should be like this：
![image](https://user-images.githubusercontent.com/42060610/179769591-9840af4a-045f-4842-a8a4-739aae93230c.png)


We next describe the parameter file.  **All the parameters of our expts can be turned in `config/sysTest.yaml` folder for replication of experimental results**. 

**Note that we take running Lazy strategy as example. The way how to run Eager is the same as Lazy. All Eager codes are  in `Eager` folder**.

```
# Here we release two datasets to their defalut parameters. see them in data/YCSB_new folder.
# YCSB name: Synthetic_WrtFreq0.5_QueryNum1000_QSizeFB_Zipfian1.2_RSize8_WSize8_Len5000
# Wiki name: Wiki_WrtFreq0.5_thresh10000_rnd0_RSize8_WSize8_Len5000

dataset_name: Wiki_WrtFreq0.5_thresh10000_rnd0_RSize8_WSize8_Len5000
dataset_root: data/datasets

ML_model_root: ML/model/
# specify the cache policy: bMCP, sMCP, oMCP, Belady (mcBelady), LRU, Belady_txn (BeladySet), LRU_txn (LRUSet), LRU_k
alg_name: 'bMCP' 
# must less than 5000
batch_size: 1000
# fix it to LCC, no need to change
cache_scheme: LCC
# cache size ratio, less than 1
csize: 0.2
dataset_name: Wiki_WrtFreq0.5_thresh10000_rnd0_RSize8_WSize8_Len5000
dataset_root: data/YCSB_new
# oldest (Lazy)
fetch_strategy: oldest
# control the accuracy of ML
ob_acc: 1
# control har far procedure OB find obsolete items.
opt_len: 5
# s
staleness_bound: 10
# Control system on or off. You must correctly install and configure the Redis(Memcached) and Hbase; otherwise
# an error will be reported. You may set it to fasle to check the #dbread result, refers to 'cost'.
sys_test: false
# the current input model, only work in ML-related task
semionline_flag: false
```

sys_test must set to be false if you do not correctly install and configure the Memcache and Hbase.

**How to run in different input models**:

**1) offline model**

We can test all the cache policies in offline model by turning the above parameters in `config/sysTest.yaml`. Some key parameters can be set as follows:

alg_name: bMCP, Belady, LRU, Belady_txn, LRU_txn, LRU_k.  

ob_acc must set to 1.

**2) semi-online model**

alg_name: sMCP, Belady, LRU, Belady_txn, LRU_txn, LRU_k.  

ob_acc can be turned from 0.75 to 1. 

If we set alg_name is sMCP and ob_acc is 1, then the result is the same as the one in offline model.

We remark when ML accuracy (i.e., ob_acc) of sMCP is controlled, it is actually bMCP.

**3) online model**

alg_name: oMCP, LRU, LRU_txn, LRU_k.  

ob_acc can be turned from 0.75 to 1. 


## ML-augmented 

We can also simply test the performance of all cache policies for semi-online and online models, especially sMCP and oMCP.

Note that all the ML files are in `ML` folder, including feature generations, lightGBM, and the parameters for lightGBM.

To train the model, run the command:
```
python3 train_ML.py
```

Based on the generated ML model, we can test the performance by:
```
python3 test_ML.py
```

**1) semi-online model**

semionline_flag = True (`config/sysTest.yaml`)

alg_name: sMCP

**2) online model**

semionline_flag = False

alg_name: oMCP

## System configurations

All the system-ralated parameters are set in `config/backend.ini`. You do no need to change them in general.

### Redis

Just run below script to start a specified port redis client.

```
bash redis-setup.sh 6380
```

And check if redis service is opening:
```
ps -aux | grep redis
```

### Memcached

We use Memcached as our cache nodes. You should install memcached server and turn on its service.

How to install in Ubuntu:

https://www.digitalocean.com/community/tutorials/how-to-install-and-secure-memcached-on-ubuntu-18-04

Check if memcaches service is opening:
```
ps -aux | grep memcached
```

### Hbase
Here we use a hbase docker deployed in AWS ECS. You may find related image: https://hub.docker.com/r/dajobe/hbase/

We also provide it in `docker/hbase-docker`.

When docker generates successfully the container, you need to enter the container and create an empty Hbase table first
```
docker exec -it hbase-docker bash
hbase shell
create 'usertable', 'family', {NAME => 'family', BLOCKCACHE => 'false'}
```
** Note that `hbase-docker` means the container name, you may change it to your computer's corresponding name **.

### AWS ECS(EC2)

Our all experiments are run on AWS: https://aws.amazon.com/

See how to employ docker image in AWS ECR: https://aws.amazon.com/ecr/

See how to create docker container in AWS ECS: https://aws.amazon.com/ecs/



## Datasets

This part just to show how to generate our datasets. You may skip it even if all the scripts has been provided.

Note we have released the two datasets to their defalut parameters for tests. see them in `data/YCSB_new folder`.

The related files are `config/genYCSB.yaml`, `config/toTCache.yaml`, `config/toYCSB.yaml`, `wiki_trace_preprocess.py`, `YCSB_workload.py`.


### **YCSB** 

   1. **Generate query_size file and transaction_size file for YCSB.**
   
      `querysize.txt`, `txnflagsize.txt` will be saved (YCSB choose specific queries for each transaction).
      
      Run `$ python YCSB_workload.py --yaml_file config/genYCSB.yaml` with parameters to specify:
   ```
   --yaml_file: yaml file containing values of all the arguments below
   --ycsb_root: str, YCSB workload parent directory
   --dataset_name: str, YCSB workload name
   --func: str, should be 'genYCSB'
   --query_num, --txn_num, --wrt_freq, --rsize, --wsize, --multi_size_qry, --max_qry_size, 
   ```
   2. **Create workload configuration file for YCSB.**

      `$YCSB_HOME/script/TCache.sh` will be executed to create a new `workload` file based on `$YCSB_HOME/script/workload_template`, "dataset name" and other variables can be provided in command line arguments or taken from user input.

   ```
   $ cd $YCSB_HOME   # go to $YCSB_HOME directory
   $ ./script/TCache.sh config   # $1=config must be specified
   ```
   3. **Load data to DB using YCSB and dump to remote server.**
   
   ```
   bin/ycsb load hbase2 -P /mnt/d/stream_cc_exp/YCSB/datasets/Synthetic_WrtFreq0.5_QueryNum1000_QSizeFB_Zipfian1.3_RSize8_WSize8_Len5000/workload -p table=usertable -p columnfamily=family -s > /mnt/d/stream_cc_exp/YCSB/datasets/Synthetic_WrtFreq0.5_QueryNum1000_QSizeFB_Zipfian1.3_RSize8_WSize8_Len5000/load.dat
   ``` 
   4. **YCSB generate transactions.**
   
   ```
   bin/ycsb run hbase2 -P /mnt/d/stream_cc_exp/YCSB/datasets/Synthetic_WrtFreq0.45_QueryNum1000_QSizeFB_Zipfian1.2_RSize8_WSize8_Len5000/workload -p table=usertable -p columnfamily=family -s > /mnt/d/stream_cc_exp/YCSB/datasets/Synthetic_WrtFreq0.45_QueryNum1000_QSizeFB_Zipfian1.2_RSize8_WSize8_Len5000/transactions.dat
   ```
   5. **Convert YCSB output to TCache Input.**
   
      First, copy YCSB output file to TCache directory if needed for system testing. 
      Run `$ python YCSB_workload.py --yaml_file config/toTCache.yaml`


### **WikiTrace**
From thie paper 'Learning relaxed belady for content distribution network caching' for original data.
And see `wiki_trace_preprocess.py` how to process the original data.
   1. Slice `.tr` file into 28 pieces, save to numpy file
   2. For each `piece` of trace, apply appearance thresh filter and perform random sampling over the filtered trace
   3. Generate TCache transaction sequence input from sampled wikitrace based on read-only and write-only transaction size (# of queries in each transaction), write frequency, transaction length.


