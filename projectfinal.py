# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:23:40 2021

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:02:14 2021

@author: ASUS
"""

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

train_data = pd.read_csv("E:/AI Project (9-2021)/Twitter_Sentiment_Analysis/train.csv")

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

def testing(text_input):
    result=(model.predict(Tfidf_vect.transform([text_input]))[0])
    if result==1:
        return result
    elif result==0:
        return result