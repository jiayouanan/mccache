#!/usr/bin/python
# -*- coding: UTF-8 -*-

# ======================================================================================
# = lgbm config
# ======================================================================================


# only valid for grid search tuning
is_tune = False

# ======================================================================================
# =  general parameters
# ======================================================================================

MAX_ROUNDS = 5000
early_stopping_rounds = 100
verbose_eval = 100
# VC num
nfold = 3

# ======================================================================================
# =  search tuning parameters
# ======================================================================================

early_stopping_rounds_grid = 20


params = {
    'boosting':'dart',
    'num_leaves': 20,
    'max_depth': 15,
    'max_bin': 255,
    'min_data_in_leaf': 300,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 2,
    'lambda_l1': 0.0,
    'lambda_l2': 0.0,
    'learning_rate': 0.1,
    'objective': 'binary', 
    'num_iterations': 2000, 
    'metric': 'binary_error', #'binary', 'binary_error','auc'
    'num_threads': 4,
    'is_unbalance': 'true'
}
# 1
num_leaves_can = [50, 100, 150, 200, 250]
max_depth_can = [7, 9,11,13]
# 2
max_bin_can = [150, 255, 299, 355,399]
min_data_in_leaf_can = [50, 150, 300]
# 3
feature_fraction_can = [0.5, 0.7, 0.9,0.99]
bagging_fraction_can = [0.5, 0.7, 0.9,0.99]
bagging_freq_can = [2, 5,8,10]
# 4
lambda_l1_can = [0.01, 0.05, 0.1]
lambda_l2_can = [0.01, 0.05, 0.1]
min_split_gain_can = [0.01, 0.1]
# 5
learning_rate_can = [0.01, 0.05,0.1]
