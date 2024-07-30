import tkinter as tk
from tkinter import scrolledtext
import random
import json
import pickle
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer and load model and data
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bagw(sentence):
    sentence_words = clean_up_sentences(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bagw(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Function to send message and get response
def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("1.0", tk.END)

    if msg != '':
        ChatLog.config(state=tk.NORMAL)
        ChatLog.insert(tk.END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        # Get chatbot response
        ints = predict_class(msg)
        res = get_response(ints, intents)

        ChatLog.insert(tk.END, "Bot: " + res + '\n\n')
        ChatLog.config(state=tk.DISABLED)
        ChatLog.yview(tk.END)

# GUI setup
base = tk.Tk()
base.title("Chatbot")
base.geometry("400x500")
base.resizable(width=False, height=False)

ChatLog = scrolledtext.ScrolledText(base, bd=0, bg="white", height="8", width="50", font="Arial")
ChatLog.config(state=tk.DISABLED)

SendButton = tk.Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5, bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff', command=send)
EntryBox = tk.Text(base, bd=0, bg="white", width="29", height="5", font="Arial")

ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
