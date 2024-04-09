import numpy as np
import pickle
import torch

transaction_features = ['mcc_code', 'currency_rk', 'transaction_amt', 'dweek', 'year', 'month',
       'hour', 'week', 'days', 'days_dif']

# transaction_features = ['currency', 'operation_kind', 'card_type', 'operation_type',
#                         'operation_type_group', 'ecommerce_flag', 'payment_system',
#                         'income_flag', 'mcc', 'country', 'city', 'mcc_category',
#                         'day_of_week', 'hour', 'weekofyear', 'amnt', 'days_before', 'hour_diff']


def batches_generator(data, batch_size=32, is_infinite=False,
                      verbose=False, device=None, output_format='torch', is_train=True):
    """
    функция для создания батчей на вход для нейронной сети для моделей на keras и pytorch.
    так же может использоваться как функция на стадии инференса
    :param list_of_paths: путь до директории с предобработанными последовательностями
    :param batch_size: размер батча
    :param shuffle: флаг, если True, то перемешивает list_of_paths и так же
    перемешивает последовательности внутри файла
    :param is_infinite: флаг, если True,  то создает бесконечный генератор батчей
    :param verbose: флаг, если True, то печатает текущий обрабатываемый файл
    :param device: device на который положить данные, если работа на торче
    :param output_format: допустимые варианты ['tf', 'torch']. Если 'torch', то возвращает словарь,
    где ключи - батчи из признаков, таргетов и app_id. Если 'tf', то возвращает картеж: лист input-ов
    для модели, и список таргетов.
    :param is_train: флаг, Если True, то для кераса вернет (X, y), где X - input-ы в модель, а y - таргеты, 
    если False, то в y будут app_id; для torch вернет словарь с ключами на device.
    :return: бачт из последовательностей и таргетов (или app_id)
    """
    while True:
        padded_sequences, targets = data['padded_sequences'], data['targets']
        user_ids = data['user_ids']
        indices = np.arange(len(user_ids))

        for idx in range(len(user_ids)):
            bucket = padded_sequences[idx]
            user_id = user_ids[idx]
            
            if is_train:
                target = targets[idx]
            
            for jdx in range(0, len(bucket), batch_size):
                batch_sequences = bucket[jdx: jdx + batch_size]
                if is_train:
                    batch_targets = target[jdx: jdx + batch_size]
                
                # batch_products = product[jdx: jdx + batch_size]
                batch_user_ids = user_id[jdx: jdx + batch_size]
                
                if output_format == 'tf':
                    batch_sequences = [batch_sequences[:, i] for i in
                                        range(len(transaction_features))]
                    
                    # append product as input to tf model
                    # batch_sequences.append(batch_products)
                    if is_train:
                        yield batch_sequences, batch_targets
                    else:
                            yield batch_sequences, batch_user_ids
                else:
                    batch_sequences = [torch.LongTensor(batch_sequences[:, i]).to(device)
                                        for i in range(len(transaction_features))]
                    if is_train:
                        yield dict(transactions_features=batch_sequences,
                                #    product=torch.LongTensor(batch_products).to(device),
                                    label=torch.LongTensor(batch_targets).to(device),
                                    user_id=batch_user_ids)
                    else:
                        yield dict(transactions_features=batch_sequences,
                                #    product=torch.LongTensor(batch_products).to(device),
                                    uesr_id=batch_user_ids)
        if not is_infinite:
            break
