import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


st.title("SMS Spam Detection")


# User input
sms = st.text_input("Enter your SMS")
stemmer = PorterStemmer()

def text_transfrom(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y.copy()
    y.clear()
    stop_words = set(stopwords.words('english'))

    for i in text:
        if i not in stop_words:
            y.append(i)

    for i in text:
        y.append(stemmer.stem(i))
    return " ".join(y)
# Button
def pre_fun(msg):
    new_msg = text_transfrom(msg)
    vect_msg = vectorizer.transform([new_msg])
    pre = model.predict(vect_msg)

    return (pre[0])




if st.button("Predict"):
    output = pre_fun(sms)
    if output==1:
        st.write("Spam")
    else:
        st.write("Not Spam")
 
