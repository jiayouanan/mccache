# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd



def generate_read_features(time_step, batch_start, batch_end, txn_id_seq, txn_item_dict, 
                           item_read_time_df, item_write_time_df, semionline=False):
    

    txn_id = txn_id_seq[time_step - batch_start]
    txn_vec = txn_item_dict[txn_id]
    
    # read features
    slice_read_time_df = item_read_time_df.loc[txn_vec,:]

    slice_read_time_df['item_id'] = slice_read_time_df.index
    slice_read_time_df['time_step'] = time_step
    
    
    slice_read_time_df['read_length'] = slice_read_time_df.iloc[:,0].apply(lambda r: len(r))
    slice_read_time_df['read_total_iten_len'] = sum(slice_read_time_df['read_length'])

    # slice_read_time_df['read_arrive_times_p1'] = slice_read_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i > time_step-1, [x for x in r if x < time_step]))))
    #slice_read_time_df['read_arrive_times_p10'] = slice_read_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i > time_step-10, [x for x in r if x < time_step]))))
    slice_read_time_df['read_arrive_times_p30'] = slice_read_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i > time_step-30, [x for x in r if x < time_step]))))
    slice_read_time_df['read_arrive_times_p50'] = slice_read_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i > time_step-50, [x for x in r if x < time_step]))))
    slice_read_time_df['read_arrive_times_p100'] = slice_read_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i > time_step-100, [x for x in r if x < time_step]))))        
    slice_read_time_df['read_arrive_times_p200'] = slice_read_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i > time_step-200, [x for x in r if x < time_step]))))        
    

    if semionline==True:
        slice_read_time_df['read_arrive_times_f1'] = slice_read_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i < time_step+1, [x for x in r if x > time_step]))))
        slice_read_time_df['read_arrive_times_f10'] = slice_read_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i < time_step+10, [x for x in r if x > time_step]))))
        slice_read_time_df['read_arrive_times_f30'] = slice_read_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i < time_step+30, [x for x in r if x > time_step]))))
        slice_read_time_df['read_arrive_times_f50'] = slice_read_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i < time_step+50, [x for x in r if x > time_step]))))
        slice_read_time_df['read_arrive_times_f100'] = slice_read_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i < time_step+100, [x for x in r if x > time_step]))))
    

                         
    slice_read_time_df['read_p_delta0'] = slice_read_time_df.iloc[:,0].apply(lambda r: time_step - [x for x in r if x < time_step][0] if len([x for x in r if x < time_step]) > 0 else 0) #np.nan
    slice_read_time_df['read_p_delta1'] = slice_read_time_df.iloc[:,0].apply(lambda r: [x for x in r if x < time_step][0] - [x for x in r if x < time_step][1] if len([x for x in r if x < time_step]) > 1 else 0)
    slice_read_time_df['read_p_delta2'] = slice_read_time_df.iloc[:,0].apply(lambda r: [x for x in r if x < time_step][1] - [x for x in r if x < time_step][2] if len([x for x in r if x < time_step]) > 2 else 0)
    slice_read_time_df['read_p_delta3'] = slice_read_time_df.iloc[:,0].apply(lambda r: [x for x in r if x < time_step][2] - [x for x in r if x < time_step][3] if len([x for x in r if x < time_step]) > 3 else 0)
    slice_read_time_df['read_p_delta4'] = slice_read_time_df.iloc[:,0].apply(lambda r: [x for x in r if x < time_step][3] - [x for x in r if x < time_step][4] if len([x for x in r if x < time_step]) > 4 else 0)
    slice_read_time_df['read_p_delta5'] = slice_read_time_df.iloc[:,0].apply(lambda r: [x for x in r if x < time_step][4] - [x for x in r if x < time_step][5] if len([x for x in r if x < time_step]) > 5 else 0)

    slice_read_time_df['read_p_delta0_delta1'] = slice_read_time_df['read_p_delta0'] - slice_read_time_df['read_p_delta1']
    slice_read_time_df['read_p_delta1_delta2'] = slice_read_time_df['read_p_delta1'] - slice_read_time_df['read_p_delta2']
    slice_read_time_df['read_p_delta2_delta3'] = slice_read_time_df['read_p_delta2'] - slice_read_time_df['read_p_delta3']
    slice_read_time_df['read_p_delta3_delta4'] = slice_read_time_df['read_p_delta3'] - slice_read_time_df['read_p_delta4']
    slice_read_time_df['read_p_delta4_delta5'] = slice_read_time_df['read_p_delta4'] - slice_read_time_df['read_p_delta5']
    
    if semionline==True:
        slice_read_time_df['read_f_delta0'] = slice_read_time_df.iloc[:,0].apply(lambda r: [x for x in r if x > time_step][-1] - time_step if len([x for x in r if x > time_step]) > 0 else 0)
        slice_read_time_df['read_f_delta1'] = slice_read_time_df.iloc[:,0].apply(lambda r: [x for x in r if x > time_step][-2] - [x for x in r if x > time_step][-1] if len([x for x in r if x > time_step]) > 1 else 0)
        slice_read_time_df['read_f_delta2'] = slice_read_time_df.iloc[:,0].apply(lambda r: [x for x in r if x > time_step][-3] - [x for x in r if x > time_step][-2] if len([x for x in r if x > time_step]) > 2 else 0)
     

    
    # write features
    slice_write_time_df = item_write_time_df.loc[txn_vec,:]
    
    slice_write_time_df['wrt_length'] = slice_write_time_df.iloc[:,0].apply(lambda r: len(r))
    slice_write_time_df['wrt_total_iten_len'] = sum(slice_write_time_df['wrt_length'])
#    slice_write_time_df['item_id'] = slice_write_time_df.index
#    slice_write_time_df['time_step'] = time_step
    
#    slice_write_time_df['write_arrive_times_p1'] = slice_write_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i > time_step-1, [x for x in r if x < time_step]))))
#    slice_write_time_df['write_arrive_times_p10'] = slice_write_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i > time_step-10, [x for x in r if x < time_step]))))
    slice_write_time_df['write_arrive_times_p30'] = slice_write_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i > time_step-30, [x for x in r if x < time_step]))))
    slice_write_time_df['write_arrive_times_p50'] = slice_write_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i > time_step-50, [x for x in r if x < time_step]))))
    slice_write_time_df['write_arrive_times_p100'] = slice_write_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i > time_step-100, [x for x in r if x < time_step]))))        
    slice_write_time_df['write_arrive_times_p200'] = slice_write_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i > time_step-200, [x for x in r if x < time_step]))))        


    if semionline==True:
        slice_write_time_df['write_arrive_times_f1'] = slice_write_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i < time_step+1, [x for x in r if x > time_step]))))
        slice_write_time_df['write_arrive_times_f10'] = slice_write_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i < time_step+10, [x for x in r if x > time_step]))))
        slice_write_time_df['write_arrive_times_f30'] = slice_write_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i < time_step+30, [x for x in r if x > time_step]))))
        slice_write_time_df['write_arrive_times_f50'] = slice_write_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i < time_step+50, [x for x in r if x > time_step]))))
        slice_write_time_df['write_arrive_times_f100'] = slice_write_time_df.iloc[:,0].apply(lambda r: np.sum(list(map(lambda i: i < time_step+100, [x for x in r if x > time_step]))))
                                

    slice_write_time_df['write_p_delta0'] = slice_write_time_df.iloc[:,0].apply(lambda r: time_step - [x for x in r if x < time_step][0] if len([x for x in r if x < time_step]) > 0 else 0)
    slice_write_time_df['write_p_delta1'] = slice_write_time_df.iloc[:,0].apply(lambda r: [x for x in r if x < time_step][0] - [x for x in r if x < time_step][1] if len([x for x in r if x < time_step]) > 1 else 0)
    slice_write_time_df['write_p_delta2'] = slice_write_time_df.iloc[:,0].apply(lambda r: [x for x in r if x < time_step][1] - [x for x in r if x < time_step][2] if len([x for x in r if x < time_step]) > 2 else 0)
    slice_write_time_df['write_p_delta3'] = slice_write_time_df.iloc[:,0].apply(lambda r: [x for x in r if x < time_step][2] - [x for x in r if x < time_step][3] if len([x for x in r if x < time_step]) > 3 else 0)
    slice_write_time_df['write_p_delta4'] = slice_write_time_df.iloc[:,0].apply(lambda r: [x for x in r if x < time_step][3] - [x for x in r if x < time_step][4] if len([x for x in r if x < time_step]) > 4 else np.nan)
    slice_write_time_df['write_p_delta5'] = slice_write_time_df.iloc[:,0].apply(lambda r: [x for x in r if x < time_step][4] - [x for x in r if x < time_step][5] if len([x for x in r if x < time_step]) > 5 else np.nan)

    slice_write_time_df['write_p_delta0_delta1'] = slice_write_time_df['write_p_delta0'] - slice_write_time_df['write_p_delta1']
    slice_write_time_df['write_p_delta1_delta2'] = slice_write_time_df['write_p_delta1'] - slice_write_time_df['write_p_delta2']
    slice_write_time_df['write_p_delta2_delta3'] = slice_write_time_df['write_p_delta2'] - slice_write_time_df['write_p_delta3']
    slice_write_time_df['write_p_delta3_delta4'] = slice_write_time_df['write_p_delta3'] - slice_write_time_df['write_p_delta4']
    slice_write_time_df['write_p_delta4_delta5'] = slice_write_time_df['write_p_delta4'] - slice_write_time_df['write_p_delta5']
    
    
    if semionline==True:
        slice_write_time_df['wrt_f_delta0'] = slice_write_time_df.iloc[:,0].apply(lambda r: [x for x in r if x > time_step][-1] - time_step if len([x for x in r if x > time_step]) > 0 else 0)
        slice_write_time_df['wrt_f_delta1'] = slice_write_time_df.iloc[:,0].apply(lambda r: [x for x in r if x > time_step][-2] - [x for x in r if x > time_step][-1] if len([x for x in r if x > time_step]) > 1 else 0)
        slice_write_time_df['wrt_f_delta2'] = slice_write_time_df.iloc[:,0].apply(lambda r: [x for x in r if x > time_step][-3] - [x for x in r if x > time_step][-2] if len([x for x in r if x > time_step]) > 2 else 0)
      
    slice_df = pd.concat([slice_read_time_df,slice_write_time_df], axis=1)
    
    return slice_df