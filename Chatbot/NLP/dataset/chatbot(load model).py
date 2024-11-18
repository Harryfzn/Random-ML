import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import random
from keras.models import load_model

# Inisialisasi lemmatizer
lemmatizer = WordNetLemmatizer()

# Memuat intents
dataset = open(r'dataset\intents.json').read()
intents = json.loads(dataset)

# (Proses pelatihan model di sini...)

# Memuat intents
dataset = open(r'dataset\intents.json').read()
intents = json.loads(dataset)

# Memuat words dan classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Memuat model
model = load_model('mymodel.h5')

# Fungsi untuk membersihkan input
def clean_up(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Fungsi untuk membuat bag of words
def create_bow(sentence, words):
    sentence_words = clean_up(sentence)
    bag = list(np.zeros(len(words)))
    
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Fungsi untuk memprediksi kelas
def predict_class(sentence, model):
    p = create_bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    threshold = 0.5
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for result in results:
        return_list.append({'intent': classes[result[0]], 'prob': str(result[1])})
    
    return return_list

# Fungsi untuk mendapatkan respons
def getResponse(ints, intents_json):
    if not ints:
        return "Maaf, saya tidak bisa menemukan jawaban untuk itu."

    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    result = "Maaf, saya tidak bisa menemukan jawaban untuk itu."  # Nilai default

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    return result

# Fungsi untuk mengatur chatbot
def chatbot_response(text):
    ints = predict_class(text, model)
    if not ints:
        return "Maaf, saya tidak bisa menemukan jawaban untuk itu."
    
    res = getResponse(ints, intents)
    return res

# Mulai chatbot
def start():
    print("Halo, ini adalah chatbot Anda.")
    end = "exit"
    while True:
        inp = str(input("You: ")).lower()
        if inp == end:
            break
        if inp == '':
            print("Kosong")
        else:
            print(f"Bot: {chatbot_response(inp)}\n")



