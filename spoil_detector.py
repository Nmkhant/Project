from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.stem import WordNetLemmatizer
import pickle
import pandas as pd
import time
import numpy as np
import re
import spacy
import nltk
import streamlit as st

pickle_in=open('finalizing_model.h5','rb')
spoil_predictor=pickle.load(pickle_in)

#label encoder
le = LabelEncoder()

def spoil_predict(text):
     # loading the dataset
     #data = pd.read_csv("Language Detection.csv")
     #y = data["Language"]
     # label encoding
     #y = le.fit_transform(y)
     #Cleaning the input text
     text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]','', text)
     text = re.sub(r'[[]]', '', text)
     text = text.lower()
     data = [text]
     # converting text to bag of words model (Vector)
     x = cv.transform(data).toarray() 
     # predicting the language
     result = spoil_predictor.predict(x)
     # finding the language corresponding the the predicted value
     #result = le.inverse_transform(lang) 
     # return the predicted language
     return result



st.sidebar.image("https://media.cgtrader.com/variants/3FTRFYn5WZCfW2VJvCmHh8EX/e44aa6a6359827c9089792cde0c079681b83d3b5c3037cc0525c25607e54355b/Penrose_Triangle-01-01.png" , width=300)

menu = ["Home" , "About Us"]

choice = st.sidebar.selectbox("Menu" , menu)

if choice == "Home": #Home
    
    st.markdown("<h1 style='text-align: center; color: #ffba00'>Spoil Reviews Detector Software that's as easy to understand as it is to use</h1>", unsafe_allow_html=True)
    
    st.text("")
    
    st.image("https://static.amazon.jobs/teams/53/images/IMDb_Header_Page.jpg?1501027252" , width = 700)
    
    st.text("")
    
    user_review = st.text_input('Enter The Review You Want To Test')
    result=""
    if st.button("Predict"):
        result = spoil_predict(user_review)
        if result == 0:
            st.write("This review is not a spoil review.")
        else:
            st.write("This review is a spoil review.")
        st.write("Well Done")
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
