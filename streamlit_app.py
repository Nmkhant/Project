import pandas as pd
import numpy as np
import streamlit as st
import re
import seaborn as sns
import string
import nltk
import warnings 
import pickle
#import twitter_sentiment_analysis_tfidf_logistic
warnings.filterwarnings("ignore", category=DeprecationWarning)


st.sidebar.image("E:\AI Project (9-2021)\Penrose_Triangle-01-01.png" , width=300)

filename = 'finalized_model.h5'
loaded_model = pickle.load(open(filename,'rb'))

def testing(text_input):
    result=(loaded_model.predict(tfidf.transform([text_input]))[0])
    if result==1:
        return result
    elif result==0:
        return result

menu = ["Home" , "About Us"]

choice = st.sidebar.selectbox("Menu" , menu)

if choice == "Home": #Home
    
    st.markdown("<h1 style='text-align: center; color: #ffba00'>Spoil Reviews Detector Software that's as easy to understand as it is to use</h1>", unsafe_allow_html=True)
    
    st.text("")
    
    st.image("E:\AI Project (9-2021)\IMDb_Header_Page.jpg" , width = 700)
    
    st.text("")
    
    user_review = st.text_input('Enter The Review You Want To Test')
    result=""
    if st.button("Predict"):
        testing(user_review)

    else:
        st.markdown("<p style= 'color:red'>Please Enter A Review</p>", unsafe_allow_html=True)

   
else: #About Us
    
    st.markdown("<p style='text-align: left; color: #ffba00; font-size: 150%'><b>We are Team Trio. We made the app together by doing our part task. Here, we         want to tell about ourself.</b></p>", unsafe_allow_html=True)
    
    st.text("")
    
    col1 , col2 , col3 = st.columns(3)
    col1.image('https://th.bing.com/th/id/OIP.W-P6hA0MFd0MfJUWtC025gAAAA?pid=ImgDet&rs=1', width = 200)
    col1.write("<p style = 'text-align: left; font-size:120%; color:#ffba00'>I am Nyi Min Khant. I am a student from UTYCC. I made the User Interface of this               software.</p>", unsafe_allow_html = True)
    
    col2.image('https://th.bing.com/th/id/OIP.W-P6hA0MFd0MfJUWtC025gAAAA?pid=ImgDet&rs=1', width = 200)
    col2.write("<p style = 'text-align: left; font-size:120%; color:#ffba00'></p>", unsafe_allow_html = True)
    
    col3.image('https://th.bing.com/th/id/OIP.W-P6hA0MFd0MfJUWtC025gAAAA?pid=ImgDet&rs=1', width = 200)  
    col3.write("<p style = 'text-align: left; font-size:120%; color:#ffba00'></p>", unsafe_allow_html = True)