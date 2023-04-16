import streamlit as st
import pickle
import sklearn
import nltk

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

import string 
# string.punctuation
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()



def transform_text(text):
  text = str(text)
  text=text.lower()
  text = nltk.word_tokenize(text)
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)
  text = y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  text = y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))            
  return " ".join(y)


st.title("Email/SMS spam classifier")

input_sms = st.text_input("Enter the message here")

if st.button('Predict'):
    # Step 1 preprocess
    transformed_sms = transform_text(input_sms)
    # step 2 vectorizer
    vector_input= tfidf.transform([transformed_sms])
    # step 3 predict
    result=model.predict(vector_input)[0]
    # step 4 Display
    if result==1:
         st.header("spam")
   
    else:
        st.header("not spam")
