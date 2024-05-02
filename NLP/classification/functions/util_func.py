# import nltk 
# nltk.download('stopwords') 

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import string
import numpy as np
import torch
import json
import time

#Преобработка данных
def data_preprocessing(text: str) -> str:
    """preprocessing string: lowercase, removing punctuation and stopwords, make stemming

    Args:
        text (str): input string for preprocessing

    Returns:
        str: preprocessed string
    """

    text = text.lower()
    # text = re.sub("<\n>", "", text)  
    text = "".join([c for c in text if c not in string.punctuation]) #Удаляем знаки препинания

    stop_words = set(stopwords.words('russian'))
    splitted_text = [word for word in text.split() if word not in stop_words] # Удаляем стоп слова ("и", "в", "на" и тп)

    stemmer = SnowballStemmer("russian")
    stemmed_words = [stemmer.stem(word) for word in splitted_text] # стеминг, приводим слова к корневой форме

    text = " ".join(stemmed_words)
    return text

#Кодировка 
def encode(x:str): 
    if x == 'Good':
        return 2
    elif x == 'Neutral':
        return 1
    else:
        return 0
    
def decode(x:int):
    if x == 2:
        return 'Good'
    elif x == 1:
        return 'Neutral'
    else:
        return 'Bad'
    
#Функция для предсказания
def predict(text:str,vectorizer, classifier):
    start_time = time.time()
    text = data_preprocessing(text)
    text = vectorizer.transform([text])
    predictions = classifier.predict(text)
    predictions = decode(predictions)
    prediction_time = round(time.time()-start_time,2)
    return predictions, prediction_time

# Функция для паддинга, у каждого отзыва своя длинна. Для передачи в нейрсеть нам нужно привести их к одному формату
def padding(review_int: list, seq_len: int, single_string = False) -> np.array:
    """Make left-sided padding for input list of tokens

    Args:
        review_int (list): input list of tokens
        seq_len (int): max length of sequence, it len(review_int[i]) > seq_len it will be trimmed, else it will be padded by zeros

    Returns:
        np.array: padded sequences
    """
    if single_string == True:
        features = np.zeros((1, seq_len), dtype=int)
        if len(review_int) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review_int)))
            new = zeros + review_int
        else:
            new = review_int[:seq_len]
        features = np.array(new)
        return features


    features = np.zeros((len(review_int), seq_len), dtype=int)
    for i, review in enumerate(review_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[:seq_len]
        features[i, :] = np.array(new)

    return features


from dataclasses import dataclass

# Создание конфига для нейросети
@dataclass
class RNN_config:
    vocab_size: int
    device : str
    n_layers : int
    embedding_dim : int
    hidden_size : int
    seq_len : int
    bidirectional : bool or int

#Функция для предсказания GRU модели
def predict_gru(text:str, path_to_model='model_gru.pt',path_to_vocab = 'vocab_for_GRU.json', SEQ_LEN = 154, )->str:
    start_time = time.time()
    model_gru = torch.load(path_to_model)
    with open(path_to_vocab) as f:
        vocab_to_int = json.load(f)
    text = data_preprocessing(text)
    tokenized_text = [vocab_to_int[word] for word in text.split() if vocab_to_int.get(word)]
    padded_text = padding(tokenized_text, SEQ_LEN, single_string=True)
    torched_text = torch.from_numpy(padded_text).type(torch.LongTensor)
    out = model_gru(torched_text.unsqueeze(0)).softmax(dim=-1).argmax().item()

    prediction_time = round(time.time()-start_time,2)

    return decode(out), prediction_time

# Функция предсказания BERT модели
def predict_bert(text:str, BERT_model, tokenizer, clf_model, MAX_LEN = 154)-> str:
    start_time = time.time()
    tokenized = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=MAX_LEN)
    padded = np.array(tokenized + [0]*(MAX_LEN-len(tokenized)))
    attention_mask = np.where(padded != 0, 1, 0)
    attention_mask = torch.LongTensor(attention_mask).unsqueeze(0)
    padded = torch.LongTensor(padded).unsqueeze(0)
    model = BERT_model
    features = []
    with torch.inference_mode():
            last_hidden_states = model(padded, attention_mask=attention_mask)
            vectors = last_hidden_states[0][:,0,:].numpy()
            features = vectors
    clf = clf_model
    out = clf.predict(features)
    prediction_time = round(time.time()-start_time,2)

    return decode(out), prediction_time