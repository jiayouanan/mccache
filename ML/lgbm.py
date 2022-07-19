#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Shuai An
"""

# ======================================================================================
# = lgbm main
# ======================================================================================


import lightgbm as lgb
import pandas as pd

from ML.lgbm_config import MAX_ROUNDS, early_stopping_rounds, verbose_eval, is_tune
from ML.lgbm_config import (params, num_leaves_can, max_depth_can, max_bin_can,
                                                         min_data_in_leaf_can, feature_fraction_can,
                                                         bagging_fraction_can, bagging_freq_can, lambda_l1_can, lambda_l2_can,
                                                         min_split_gain_can, learning_rate_can, early_stopping_rounds_grid, nfold)



class LgbmPreClass(object):

    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val


    def train_pre(self):
        
        dtrain = lgb.Dataset(
            self.X_train, label=self.y_train, feature_name=list(self.X_train.columns))

        dval = lgb.Dataset(
            self.X_val, label=self.y_val, reference=dtrain, 
            feature_name=list(self.X_train.columns))

        params = self.tune_parametersBygrid(dtrain, is_tune)
        #print(params)
        bst = lgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS,
            valid_sets=[dtrain, dval], early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval # verbose_eval=0
        )


        return bst

    # grid search turning parameters
    def tune_parametersBygrid(self, lgb_train, is_tune):

        
        print('set parameters')

        if is_tune:
            print("=" * 50)
            print('Starting CV!')

            print("1: tuning num_leaves and max_depth")
            print("=" * 50)
            min_merror = float('Inf')
            max_merror = -float('Inf')
            for num_leaves in num_leaves_can:
                for max_depth in max_depth_can:
                    params['num_leaves'] = num_leaves
                    params['max_depth'] = max_depth

                    cv_results = lgb.cv(
                        params,
                        lgb_train,
                        seed=42,
                        nfold=nfold,
                        metrics=['binary_error'], # binary_logloss, auc, binary_error
                        early_stopping_rounds=early_stopping_rounds_grid,
                        stratified=False
                    )  # stratified=False is important
#                    print(cv_results)
#                    mean_merror = pd.Series(cv_results['auc-mean']).max()
#
#                    if mean_merror > max_merror:
#                        max_merror = mean_merror
#                        best_num_leaves = num_leaves
#                        best_max_depth = max_depth
                        
                    mean_merror = pd.Series(cv_results['binary_error-mean']).min()

                    if mean_merror < min_merror:
                        min_merror = mean_merror
                        best_num_leaves = num_leaves
                        best_max_depth = max_depth

            params['num_leaves'] = best_num_leaves
            params['max_depth'] = best_max_depth



            # in order to void overfit
            print("=" * 50)
            print("2: tuning max_bin and min_data_in_leaf")
            print("=" * 50)
            min_merror = float('Inf')
            max_merror = -float('Inf')
            for max_bin in max_bin_can:
                for min_data_in_leaf in min_data_in_leaf_can:
                    params['max_bin'] = max_bin
                    params['min_data_in_leaf'] = min_data_in_leaf

                    cv_results = lgb.cv(
                        params,
                        lgb_train,
                        seed=42,
                        nfold=nfold,
                        metrics=['binary_error'],
                        early_stopping_rounds=early_stopping_rounds_grid,
                        stratified=False
                    )
                    
#                    mean_merror = pd.Series(cv_results['auc-mean']).max()
#
#                    if mean_merror > max_merror:
#                        max_merror = mean_merror
#                        best_max_bin = max_bin
#                        best_min_data_in_leaf = min_data_in_leaf
                        
                    mean_merror = pd.Series(cv_results['binary_error-mean']).min()

                    if mean_merror < min_merror:
                        min_merror = mean_merror
                        best_max_bin = max_bin
                        best_min_data_in_leaf = min_data_in_leaf

            params['min_data_in_leaf'] = best_max_bin
            params['max_bin'] = best_min_data_in_leaf

            print("=" * 50)
            print("3: tuning feature_fraction, bagging_fraction and bagging_freq")
            print("=" * 50)
            min_merror = float('Inf')
            max_merror = -float('Inf')
            for feature_fraction in feature_fraction_can:
                for bagging_fraction in bagging_fraction_can:
                    for bagging_freq in bagging_freq_can:
                        params['feature_fraction'] = feature_fraction
                        params['bagging_fraction'] = bagging_fraction
                        params['bagging_freq'] = bagging_freq

                        cv_results = lgb.cv(
                            params,
                            lgb_train,
                            seed=42,
                            nfold=nfold,
                            metrics=['binary_error'],
                            early_stopping_rounds=early_stopping_rounds_grid,
                            stratified=False
                        )


                            
                        mean_merror = pd.Series(cv_results['binary_error-mean']).min()
                        
                        if mean_merror < min_merror:
                            min_merror = mean_merror
                            best_feature_fraction = feature_fraction
                            best_bagging_fraction = bagging_fraction
                            best_bagging_freq = bagging_freq

#                        mean_merror = pd.Series(cv_results['auc-mean']).max()
#
#                        if mean_merror > max_merror:
#                            max_merror = mean_merror
#                            best_feature_fraction = feature_fraction
#                            best_bagging_fraction = bagging_fraction
#                            best_bagging_freq = bagging_freq
                            
            params['feature_fraction'] = best_feature_fraction
            params['bagging_fraction'] = best_bagging_fraction
            params['bagging_freq'] = best_bagging_freq

            print("=" * 50)
            print("4: tuning lambda_l1, lambda_l2 and min_split_gain")
            print("=" * 50)
            min_merror = float('Inf')
            max_merror = -float('Inf')
            for lambda_l1 in lambda_l1_can:
                for lambda_l2 in lambda_l2_can:
                    for min_split_gain in min_split_gain_can:
                        params['lambda_l1'] = lambda_l1
                        params['lambda_l2'] = lambda_l2
                        params['min_split_gain'] = min_split_gain

                        cv_results = lgb.cv(
                            params,
                            lgb_train,
                            seed=42,
                            nfold=nfold,
                            metrics=['binary_error'],
                            early_stopping_rounds=early_stopping_rounds_grid,
                            stratified=False
                        )

#                        mean_merror = pd.Series(cv_results['auc-mean']).max()
#
#                        if mean_merror > max_merror:
#                            max_merror = mean_merror
#                            best_lambda_l1 = lambda_l1
#                            best_lambda_l2 = lambda_l2
#                            best_min_split_gain = min_split_gain

                        mean_merror = pd.Series(cv_results['binary_error-mean']).min()

                        if mean_merror < min_merror:
                            min_merror = mean_merror
                            best_lambda_l1 = lambda_l1
                            best_lambda_l2 = lambda_l2
                            best_min_split_gain = min_split_gain

            params['lambda_l1'] = best_lambda_l1
            params['lambda_l2'] = best_lambda_l2
            params['min_split_gain'] = best_min_split_gain

            print("=" * 50)
            print("5: tuning learning_rate")
            print("=" * 50)
            min_merror = float('Inf')
            max_merror = -float('Inf')
            for learning_rate in learning_rate_can:
                params['learning_rate'] = learning_rate

                cv_results = lgb.cv(
                    params,
                    lgb_train,
                    seed=42,
                    nfold=nfold,
                    metrics=['binary_error'],
                    early_stopping_rounds=early_stopping_rounds_grid,
                    stratified=False
                )
                
#                mean_merror = pd.Series(cv_results['auc-mean']).max()
#
#                if mean_merror > max_merror:
#                    max_merror = mean_merror
#                    best_learning_rate = learning_rate
                    
                mean_merror = pd.Series(cv_results['binary_error-mean']).min()

                if mean_merror < min_merror:
                    min_merror = mean_merror
                    best_learning_rate = learning_rate

            params['learning_rate'] = best_learning_rate

        return params
