from typing import Dict

import numpy as np
import pandas as pd
import pickle
from tqdm.notebook import tqdm

features = ['mcc_code', 'currency_rk', 'transaction_amt', 'dweek', 'year', 'month',
       'hour', 'week', 'days', 'days_dif']

# features = ['currency', 'operation_kind', 'card_type', 'operation_type', 'operation_type_group', 'ecommerce_flag',
#             'payment_system', 'income_flag', 'mcc', 'country', 'city', 'mcc_category', 'day_of_week',
#             'hour', 'weekofyear', 'amnt', 'days_before', 'hour_diff']


def transform_transactions_to_sequences(transactions_frame: pd.DataFrame,
                                        num_last_transactions=186) -> pd.DataFrame:
    """
    принимает frame с транзакциями клиентов, сортирует транзакции по клиентам
    (внутри клиента сортирует транзакции по возрастанию), берет num_last_transactions танзакций,
    возвращает новый pd.DataFrame с двумя колонками: app_id и sequences.
    каждое значение в колонке sequences - это список списков.
    каждый список - значение одного конкретного признака во всех клиентских транзакциях.
    Всего признаков len(features), поэтому будет len(features) списков.
    Данная функция крайне полезна для подготовки датасета для работы с нейронными сетями.
    :param transactions_frame: фрейм с транзакциями клиентов
    :param num_last_transactions: количество транзакций клиента, которые будут рассмотрены
    :return: pd.DataFrame из двух колонок (app_id, sequences)
    """
    for feature in transactions_frame[features]:
        transactions_frame[feature] = transactions_frame[feature].astype(int)
    return transactions_frame \
        .sort_values(['user_id', 'transaction_dttm']) \
        .groupby(['user_id'])[features] \
        .apply(lambda x: truncate(x, num_last_transactions=num_last_transactions)) \
        .reset_index().rename(columns={0: 'sequences'})

def truncate(x, num_last_transactions=186):
    return x.values.transpose()[:, -num_last_transactions:].tolist()



def create_padded_buckets(frame_of_sequences: pd.DataFrame, bucket_info: Dict[int, int],
                          save_to_file_path=None, has_target=True):
    """
    Функция реализует sequence_bucketing технику для обучения нейронных сетей.
    Принимает на вход frame_of_sequences (результат работы функции transform_transactions_to_sequences),
    словарь bucket_info, где для последовательности каждой длины указано, до какой максимальной длины нужно делать
    padding, далее группирует транзакции по бакетам (на основе длины), делает padding транзакций и сохраняет результат
    в pickle файл, если нужно
    :param frame_of_sequences: pd.DataFrame c транзакциями (результат применения transform_transactions_to_sequences)
    :param bucket_info: словарь, где для последовательности каждой длины указано, до какой максимальной длины нужно делать
    padding
    :param save_to_file_path: опциональный путь до файла, куда нужно сохранить результат
    :param has_target: флаг, есть ли в frame_of_sequences целевая переменная или нет. Если есть, то
    будет записано в результат
    :return: возвращает словарь с следюущими ключами (padded_sequences, targets, app_id, products)
    """
    frame_of_sequences['bucket_idx'] = 186 #frame_of_sequences.sequence_length.map(bucket_info)
    padded_seq = []
    targets = []
    user_ids = []
    products = []

    for size, bucket in tqdm(frame_of_sequences.groupby('bucket_idx'), desc='Extracting buckets'):
        padded_sequences = bucket.sequences.apply(lambda x: pad_sequence(x, size)).values
        padded_sequences = np.array([np.array(x) for x in padded_sequences])
        padded_seq.append(padded_sequences)

        if has_target:
            targets.append(bucket.target.values)

        user_ids.append(bucket.user_id.values)
        # products.append(bucket['product'].values)

    frame_of_sequences.drop(columns=['bucket_idx'], inplace=True)

    dict_result = {
        'padded_sequences': np.array(padded_seq),
        'targets': np.array(targets) if targets else [],
        'user_ids': np.array(user_ids),
        # 'products': np.array(products),
    }

    if save_to_file_path:
        with open(save_to_file_path, 'wb') as f:
            pickle.dump(dict_result, f)
    return dict_result

def pad_sequence(array, max_len) -> np.array:
    """
    принимает список списков (array) и делает padding каждого вложенного списка до max_len
    :param array: список списков
    :param max_len: максимальная длина до которой нужно сделать padding
    :return: np.array после padding каждого вложенного списка до одинаковой длины
    """
    add_zeros = max_len - len(array[0])
    return np.array([[0] * add_zeros + list(x)  for x in array])