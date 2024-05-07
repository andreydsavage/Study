import streamlit as st
import pandas as pd

from toxicity import text2toxicity


def main():
    st.title('Определние нежелательного контента')
    st.write('Я могу находить непристойные комментарии и классифицировать их по классам')
    text = st.text_input("Напишите свой комментарий на русском языке и я его классифицирую")
    toxic =text2toxicity.text2toxicity(text, aggregate=True)
    df = pd.DataFrame(text2toxicity.text2toxicity(text, False),index=['non_toxic','insult', 'obscenity', 'threat','dangerous'], columns=["Probability"])
    st.bar_chart(df)
    st.write(f'# Вероятность, что ваш комментарий токсичный или может навредить вашей репутации {toxic:.2f}')

if __name__ == '__main__':
    main()