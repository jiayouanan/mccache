YCSB for TCache
====================================

Links
-----
Original YCSB on GitHub: [YCSB repository](https://travis-ci.org/brianfrankcooper/YCSB)

Getting Started
---------------
Building from source requires Maven 3.

To build the full distribution, with all database bindings:

    mvn clean package

To build a single database binding:

    mvn -T 1C -pl site.ycsb:hbase2-binding -DskipTests -am clean package

    mvn -pl site.ycsb:core -am package

    mvn -pl site.ycsb:postgrenosql-binding -am clean package


Setup Database System for Testing
---------------------------------
If no database binding is specified, YCSB use a simple interface layer (site.ycsb.BasicDB) as an alternative. Operations are printed to System.out


Prepare Workload Files
----------------------
To Generate a YCSB format workload, check existing ones in datasets/

    Before Generation:
    workload        // YCSB workload configuration
    querysize.txt   // for multi-size query workload
    txnflagsize.txt // specify write/read, size/query keys for each transaction
    // r'<write-flag 0/1>\t<transaction size>' for non-synthetic workload
    // r'<write-flag 0/1>\t<key_num>[,<key_num>]*' for synthetic workload

    After Generation:
    load.dat        // check database load status
    transactions.dat    // specific queries and keys for each transaction



Configure the Workload
----------------------
- Core Workloads: 6 workloads comes with original YCSB
```
workload/workloada: update heavy workload (50/50 reads and writes)
workload/workloadb: read mostly workload (95/5 reads writes)
workload/workloadc: read only (100% read)
worload/workloadd: read latest workload (new records are inserted, the most recently inserted records are the most popular)
workload/workloade: short ranges (short ranges of records are queried, instead of individual records)
workload/workloadf: read-modify-write (the client will read a record, modify it, and write back the changes)
```

- TCache Related Configurations
```
recordcount=1000    // rows in 'usertable'
fieldcount=1664     // multi-size query maximum size (columns), 1664 for python psycopg2 library limitation
fieldlength=10      // control query actual size

requestdistribution=zipfian
zipfianconstant=0.99

DB_PRINT_EXE=print_only    // ('print_only': only print queries when generating workload, any other configuration: only execute without printing)

syntheticprop=false // synthetic workloads specify query keys for each transaction outside YCSB
rwonlytxnprop=true  //(read-only, write-only transactions)
multisizeprop=true  //(use multisize queries)
loadfromdbprop=false    // not supported for now

printkeynum=true    // print query key number for TCache testing

postgrenosql.*=*    // PostgreSQL connection

dataset_dir=~/YCSB/datasets/workload_name/ // YCSB datasets directory
querysizefile=querysize.txt
txnsizefile=txnflagsize.txt
```


Running Tests
--------------------------
- load table (records) to database layer
```
load data to site.ycsb.BasicDB:
./bin/ycsb load basic -P workloads/workloada -s > load.dat
```
- run transactions
```
run transactions using site.ycsb.BasicDB:
./bin/ycsb run basic -P workloads/workloada -s > transactions.dat
```


Useful Scripts
--------------
Checkout `script/TCache.sh` when testing with PostgreSQL or HBase 2.x


Appendix & Note
-----------------
### Working with PostgreSQL
1. PostgreSQL connection info in `postgrenosql/conf/postgrenosql.properties`, default database name is `ycsb`, also, `autocommit` property is now disabled for read-only and write-only transactions.
2. In `workload` parameter file, `DB_PRINT_EXE` can be set to `print_only` to print transactions (queries) without execution or `execute` to commit transactions without printing.
3. For each read-only transaction, there is a predefined `ArrayList<TCacheQuery> readlist` containing `key` and `fields` for each query. In `postgrenosql/src/main/java/site/ycsb/postgrenosql/PostgreNoSQLDBClient.java`, the method `readtxn` is responsible for preparing those queries for print output or execution. Write-only transactions are handled likewise.
4. In `core/src/main/java/site/ycsb/DBWrapper.java`, two methods (`readtxn`, `writetxn`) are added to support read-only and write-only transactions for each "real" DB. The `DBWrapper` class is a wrapper around a "real" DB that measures latencies and counts return codes.
### Working with HBase 2.x
1. HBase Connection info in `script/conf/hbase-site.xml`
2. Modifications made in `hbase2/src/main/java/site/ycsb/hbase2/HBaseClient2.java` and `hbase2/pom.xml`
3. Checkout `hbase2/README.md` for original YCSB features (same as PostgreSQL)
