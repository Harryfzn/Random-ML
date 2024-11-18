import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import random

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation, Dropout
from tensorflow.keras.optimizers import SGD

from keras.models import load_model

import nltk
print(nltk.data.find('tokenizers/punkt'))

import os
print(os.path.exists('C:/Users/WORK PLUS/AppData/Roaming/nltk_data/tokenizers/punkt'))

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

words=[]
classes=[]
documents=[]
ignore=['?','!',',',"'s"]

dataset = open(r'dataset\intents.json').read()
intents = json.loads(dataset)


for intent in intents['intents']:
    for pattern in intent['patterns']:
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w,intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


#lemmatize
nltk.download('wordnet')

words=[lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore]
words=sorted(list(set(words)))
classes=sorted(list(set(classes)))

#pickle dump
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training=[]
output_empty=[0]*len(classes)

for doc in documents:
    bag=[]
    pattern=doc[0]
    pattern=[ lemmatizer.lemmatize(word.lower()) for word in pattern ]
    
    for word in words:
        if word in pattern:
            bag.append(1)
        else:
            bag.append(0)
    output_row=list(output_empty)
    output_row[classes.index(doc[1])]=1
    
    training.append([bag,output_row])
    
random.shuffle(training)
training=np.array(training, dtype=object)  
X_train=list(training[:,0])
y_train=list(training[:,1]) 

#Model
model=Sequential()
model.add(Dense(128,activation='relu',input_shape=(len(X_train[0]),)))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]),activation='softmax'))

#learning rate
SGD= SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam=keras.optimizers.Adam(0.001)


model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
#model.fit(np.array(X_train),np.array(y_train),epochs=200,batch_size=10,verbose=1)
weights=model.fit(np.array(X_train),np.array(y_train),epochs=200,batch_size=10,verbose=1)    
model.save('mymodel.h5',weights)


def clean_up(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[ lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def create_bow(sentence,words):
    sentence_words=clean_up(sentence)
    bag=list(np.zeros(len(words)))
    
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = create_bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    print(f"Model Prediction: {res}")  # For debugging
    threshold = 0.5
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for result in results:
        return_list.append({'intent': classes[result[0]], 'prob': str(result[1])})
    
    return return_list




def getResponse(ints, intents_json):
    if not ints:  # Jika `ints` kosong, kembalikan respons default
        return "Maaf, saya tidak bisa menemukan jawaban untuk itu."

    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    result = "Maaf, saya tidak bisa menemukan jawaban untuk itu."  # Nilai default

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break  # Hentikan loop setelah menemuthakan tag yang cocok

    return result

    
def chatbot_response(text):
    ints = predict_class(text, model)
    if not ints:  # Tambahkan cek apakah `ints` kosong
        return "Maaf, saya tidak bisa menemukan jawaban untuk itu."
    
    res = getResponse(ints, intents)
    return res
 


def start():
    print("halo this is your bot")
    end = "exit"  # Atur kata untuk keluar dari loop
    while True:
        inp = str(input("You: ")).lower()  # Tambahkan prompt agar pengguna tahu bahwa input diperlukan
        if inp == end:
            break
        if inp == '':
            print("kosong")
        else:
            print(f"bot: {chatbot_response(inp)}" + '\n')

start()