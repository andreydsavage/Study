import streamlit as st
import json
import joblib

import sys
sys.path.append("classification/functions/")
sys.path.append("classification/")

from classification.functions.util_func import predict,predict_gru, predict_bert




def main():
    # st.set_page_config(page_title="My Page Title")
    
    st.title('NLP. Review classification')

    text = st.text_input("Напишите свой отзыв на русском языке и я его классифицирую")
    if text:
        st.write(f'Ваш отзыв: {text}')

        with open('classification/models_f1_scores.json', 'r') as f:
            scores = json.load(f)

        col1, col2, col3 = st.columns(3)

        #TF-IDF    
        tf_idf_vect = joblib.load("classification/vectorizer_tfidf.pkl")
        tf_idf_clf = joblib.load("classification/MultinomialNB.pkl")
        prediction, pred_time = predict(text=text, vectorizer=tf_idf_vect, classifier=tf_idf_clf)

        col1.write(f"## TF-IDF")
        col1.write(f'Ваш отзыв классифицирован как {prediction}, за время = {pred_time} сек.')

        #GRU
        prediction_GRU, pred_time_GRU = predict_gru(text=text,conf_path = 'classification/conf_GRU.pkl',path_to_state_dict='classification/model_gru.pt',path_to_vocab = 'classification/vocab_for_GRU.json', SEQ_LEN = 154)
        col2.write(f"## GRU")
        col2.write(f'Ваш отзыв классифицирован как {prediction_GRU}, за время = {pred_time_GRU} сек.')

        #BERT
        prediction_bert, pred_time_bert = predict_bert(text = text, clf_path="classification/LogReg_Bert.pkl")
        col3.write(f"## LogReg_Bert")
        col3.write(f'Ваш отзыв классифицирован как {prediction_bert}, за время = {pred_time_bert} сек.')

        st.write('')
        st.write('### Сравнительный график метрики f1_score(macro)')
        st.bar_chart(scores)
        st.write('Так как классы не сбалансированны был использован undersampling в TF-IDF, в остальные моделях данный подавались "как есть"')
              


if __name__ == '__main__':
    main()