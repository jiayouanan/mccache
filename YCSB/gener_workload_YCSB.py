import yaml
import subprocess
import os
import csv

'''

'''     
            
para_combine= [
# [0.6, 0.25, 1000],
# [0.4, 0.5, 1000],
# [0.6, 0.5, 1000],
# [0.8, 0.5, 1000],
# [0.99, 0.5, 1000],
# [1.2, 0.5, 1000],
# [0.4, 0.05, 1000],
# [0.6, 0.05, 1000],
# [0.8, 0.05, 1000],
# [0.99, 0.05, 1000],
# [1.2, 0.05, 1000],
 
# [0.6, 0.25, 600],
# [0.6, 0.05, 600],
# [0.6, 0.1, 600],
# [0.6, 0.15, 600],
# [0.6, 0.2, 600],
# [0.4, 0.25, 600],
# [0.8, 0.25, 600],
# [0.99, 0.25, 600],
# [1.2, 0.25, 600],
# [0.6, 0.25, 200],
# [0.6, 0.25, 400],
# [0.6, 0.25, 800],

#  [0.4, 0.1, 600],
  [0.8, 0.1, 600],
  [0.99, 0.1, 600],
  [1.2, 0.1, 600],
  [0.4, 0.05, 600],
  [0.8, 0.05, 600],
  [0.99, 0.05, 600],
  [1.2, 0.05, 600]
]

'''
PART 2
Loop execute YCSB run
'''
num = 1
for para in para_combine:
    print("*"*100)
    print(num)
    print("*"*100)
#    cmd = 'cd /mnt/d/cc_exp/YCSB/'
#    status = subprocess.call(cmd, shell=True)
    data_dir = "Synthetic_WrtFreq"+str(para[1])+"_QueryNum"+str(para[2])+ \
    "_QSizeFB_Zipfian"+str(para[0])+"_RSize8_WSize8_Len5000"
    cmd = 'bin/ycsb run hbase2 -P /mnt/d/cc_exp/YCSB/datasets/'+ data_dir+ \
    '/workload -p table=usertable -p columnfamily=family -s > /mnt/d/cc_exp/YCSB/datasets/' + data_dir+ '/transactions.dat'   
    print(cmd)
    status = subprocess.call(cmd, shell=True)
    num += 1


        





    
