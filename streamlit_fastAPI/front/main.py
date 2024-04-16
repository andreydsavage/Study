import requests
import streamlit as st
import json

def main():
    st.title('Image classification')

    image = st.file_uploader('Chosse an image')


    if st.button("Classify!"):
        if image is not None:
            files= {"file": image.getvalue()}
            res = requests.post("http://0.0.0.0:3400/classify", files=files)
            st.write(json.loads(res.text)['prediction'])

if __name__ == '__main__':
    main()