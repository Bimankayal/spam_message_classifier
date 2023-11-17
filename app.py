import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()

def transform_text(text):
  text= text.lower()#lowercase
  text = nltk.word_tokenize(text)#tokenize in words
  alnum_text=[]
  for i in text:
    if i.isalnum():#removing spacial characters
        alnum_text.append(i) #alphabet numerical character append remove other charactes
  text= alnum_text[:]
  final_text =[]
  for i in alnum_text:#remove stopwords and punctuations
    if i not in stopwords.words('english') and i not in string.punctuation:
      final_text.append(i)
  text.clear()
  for i in final_text:#stemming
    text.append(ps.stem(i))
  return  " ".join(text)
tfidf = pickle.load(open('vectorizer.pkl' ,'rb'))
model = pickle.load(open('model.pkl' ,'rb'))

st.title("Email/SMS Classifier ")
input_sms= st.text_input("Enter the message ")
if st.button("Predict"):
  transform_sms= transform_text(input_sms)
  vector_input =tfidf.transform([transform_sms])
  result = model.predict(vector_input)[0]
  if result==1 :
    st.header('Spam')
  else:
    st.header("Not Spam")
  


