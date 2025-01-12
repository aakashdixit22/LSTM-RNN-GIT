#streamlit web app integration
import streamlit as st
import numpy 
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load the model
model = load_model('next_word_lstm.h5')
#load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# function to predict the next word
def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_sequence_len:
        token_list=token_list[:max_sequence_len-1]
    token_list=pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicted=model.predict(token_list,verbose=0)
    predicted_word_index=numpy.argmax(predicted)
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None

#streamlt app
st.title('Next Word Prediction With LSTM')
input_text=st.text_input('Enter the text','I am')
st.button('Predict Next Word')
if st.button:
    predicted_word=predict_next_word(model,tokenizer,input_text,10)
    st.write('The next word is:',predicted_word)

