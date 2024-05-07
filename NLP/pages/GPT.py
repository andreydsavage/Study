import streamlit as st
from gpt import gpt2small

def main():
    st.title('GPT2.Text generation')

    text = st.text_input("Напишите свой и текст и я его продолжу")
    if text:
        st.write(f'Ваш отзыв: {text}')
        temperature = st.slider('Выберите температуру. Она отвечает за степень случайности следующего слова', 0, 20)
        length = st.slider('Выберите максимальную длинну генерируемого теста', 0, 100)
        top_k = st.slider('Зануляем все вероятности кроме "k" самых вероятных', 1, 20)
        top_p = st.slider('Оставляем такой минимальный сет токенов, чтобы сумма их вероятностей была не больше "p"', 0, 100)
        st.write(gpt2small.make_generation(text, temperature=temperature, top_k=top_k,top_p=top_p/100,max_length=length))

if __name__ == '__main__':
    main()
