import pandas as pd
import numpy as np
import streamlit as st
import re
import seaborn as sns
import string
import nltk
import warnings 
import pickle
warnings.filterwarnings("ignore", category=DeprecationWarning)

import nltk
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('punkt')

lemmatizer=WordNetLemmatizer()
stemming = PorterStemmer()
stops = set(stopwords.words("english"))

train_data = pd.read_csv("E:/AI Project (9-2021)/Dector/train.csv")

train_original = train_data.copy()

train_original.tweet = train_original.tweet.astype(str)

def apply_cleaning_function_to_list(X):
    cleaned_X = []
    for element in X:
        cleaned_X.append(clean_text(element))
    return cleaned_X

def clean_text(raw_text):
     # Convert to lower case
    text = raw_text.lower()
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Keep only words (removes punctuation + numbers)
    # use .isalnum to keep also numbers
    token_words = [w for w in tokens if w.isalpha()]
    
    # Stemming
    #stemmed_words = [stemming.stem(w) for w in token_words]
    # lemmatizing
    lemmatized_words=[lemmatizer.lemmatize(word) for word in token_words]
    
    # Remove stop words
    meaningful_words = [w for w in lemmatized_words if not w in stops]
    # Rejoin meaningful stemmed words
    joined_words = ( " ".join(meaningful_words))
    
    # Return cleaned data
    return joined_words

text_to_clean =list(train_original['tweet'])
cleaned_text = apply_cleaning_function_to_list(text_to_clean)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=CountVectorizer()
vectorizer.fit(cleaned_text)

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test =train_test_split(cleaned_text,train_original['label'],train_size=0.75,test_size=0.25,random_state=42,shuffle=True) 

Tfidf_vect=TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(cleaned_text)
Train_X_Tfidf=Tfidf_vect.transform(X_train)
Test_X_Tfidf=Tfidf_vect.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(Train_X_Tfidf,y_train)

import pickle
modelsave = 'final_model.h5'
pickle.dump(model, open(modelsave, 'wb')) 


filename = 'final_model.h5'
loaded_model = pickle.load(open(filename,'rb'))

def testing(text_input):
    result=(loaded_model.predict(Tfidf_vect.transform([text_input]))[0])
    if result==1:
        return result
    elif result==0:
        return result

    
st.sidebar.image("E:/AI Project (9-2021)/Penrose_Triangle-01-01.png" , width=300)

menu = ["Home" , "About Us"]

choice = st.sidebar.selectbox("Menu" , menu)

if choice == "Home": #Home
    
    st.markdown("<h1 style='text-align: center; color: #ffba00'>Spoil Reviews Detector Software that's as easy to understand as it is to use</h1>", unsafe_allow_html=True)
    
    st.text("")
    
    st.image("E:/AI Project (9-2021)/IMDb_Header_Page.jpg" , width = 700)
    
    st.text("")
    
    user_review = st.text_input('Enter The Review You Want To Test')
    
    if st.button("Predict"):
        prediction = testing(user_review)
        if prediction == 0:
            st.write('Positive')
        elif prediction == 1:
            st.write('Negative')

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
