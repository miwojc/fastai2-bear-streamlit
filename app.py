from fastai2.learner import load_learner
from fastai2.learner import Learner
from fastai2.vision.core import PILImage
import streamlit as st
import os
import time
import requests

# App title
st.title("The Amazing Bear Classifier!")

def predict(img):

    # Display the test image
    st.image(img, use_column_width=True)

    # Temporarily displays a message while executing 
    with st.spinner('Wait for it...'):
        time.sleep(3)

    # Load model and make prediction
    learn_inf = load_learner('export.pkl')
    pred,pred_idx,probs = learn_inf.predict(img)
    st.success(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')

# Image source selection
option = st.radio('', ['Choose a test image', 'Choose your own image'])

if option == 'Choose a test image':

    # Test image selection
    test_images = os.listdir('model/data/test/')
    test_image = st.selectbox('Please select a test image:', test_images)

    # Read the image
    file_path = 'model/data/test/' + test_image
    img = PILImage.create(file_path)
    predict(img)

if option == 'Choose your own image':
    url = st.text_input("Please input a url:")

    if url != "":
        try:
            response = requests.get(url, stream=True)
            response.raw.decode_content = True
            img = PILImage.create(response.raw)

        except:
            st.text("Invalid url!")

        predict(img)
