"""
Train model.
"""

from copy import copy

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from sklearn.model_selection import train_test_split

from eval import calc_metrics, make_predictions
# from sample import sample_negative_examples

 
def training_with_resampling(test, train_df, submit_df, catboost_params,
                             sample_size, validation=False, resample_freq=1000, 
                             random_state=42, thread_count=-1, plot=False, verbose=False):

    catboost_params  = copy(catboost_params)
    num_iterations = int(np.ceil(catboost_params['iterations']/resample_freq)) - 1
    catboost_params['iterations'] = resample_freq

    if validation:
        X_train, X_valid, y_train, y_valid = train_test_split(train_df.iloc[:,1:-2], train_df['target'].to_numpy(), test_size=0.25, random_state=1)

    else:
        y_train = train_df.target.values
        # queries_train = train2.bank.values
        X_train = train_df.iloc[:,1:-2].values

    cat_cols = ['employee_count_nm', 'bankemplstatus', 'customer_age', 'report']
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_cols)
    if validation:
        validation_pool = Pool(data=X_valid, label=y_valid, cat_features=cat_cols)
        print('train shape', X_train.shape, 'validation shape', X_valid.shape)
    else:
        print('train shape', X_train.shape)

    clf = CatBoostClassifier(thread_count=thread_count, **catboost_params)
    if validation:
        clf.fit(train_pool, eval_set=validation_pool, plot=plot, verbose=verbose, cat_features=cat_cols)
    else:
        clf.fit(train_pool, plot=plot, verbose=verbose, cat_features=cat_cols)

    for i in range(num_iterations):

        print(f'Iteration {i+1}')

        X_train = train_df.iloc[:,1:-2].values
        y_train = train_df.target.values

        train_pool = Pool(data=X_train, label=y_train, cat_features=cat_cols)
        if validation:
            print('train shape', X_train.shape, 'validation shape', X_valid.shape)
        else:
            print('train shape', X_train.shape)
        
        if validation:
            clf.fit(train_pool, eval_set=validation_pool, plot=plot,
                    verbose=verbose, init_model=clf, cat_features=cat_cols)
        else:
            clf.fit(train_pool, plot=plot, verbose=verbose, init_model=clf, cat_features=cat_cols)
    
    if test is None:
        return clf
    else:
        preds = make_predictions(clf, test, df_trans,)
        r1, mrr, precision = calc_metrics(preds, test)
        metrics = {'r1': r1, 'mrr': mrr, 'precision': precision,
                   'best_iteration': clf.best_iteration_}

        return clf, metrics, preds
