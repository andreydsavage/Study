""""
Run model training.
"""

import pickle

import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from preprocess import click_preprocess, trans_preprocess
from train import training_with_resampling


@hydra.main(config_path='config', config_name='config')
def main(config):

    print(OmegaConf.to_yaml(config))

    with open(to_absolute_path("data/transactions.pkl"), "rb") as file_:
        transactions_data = pickle.load(file_)


    matching = pd.read_csv(to_absolute_path('data/train_matching.csv'))
    matching['target'] = 1


    df_trans = trans_preprocess(transactions_data[config.trans_data], **config.trans_params)
    print(df_trans.shape)

    for feature_group in config.trans_time_features:
        df_trans = df_trans.join(transactions_data[feature_group])
    print('df_trans', df_trans.shape)


    clf = training_with_resampling(
        matching, test=None, df_trans=df_trans,
        catboost_params=config.catboost_params, **config.train_params)

    clf.save_model(to_absolute_path(f'submit/data/model_{config.run_number}.cbm'))
    trans_filename = f'submit/data/trans_features_{config.run_number}.pkl'

    with open(to_absolute_path(trans_filename), 'wb') as file_:
        pickle.dump(df_trans.columns.tolist(), file_)


if __name__ == '__main__':

    main()
