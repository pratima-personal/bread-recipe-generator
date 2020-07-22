#!/usr/bin/env python3

import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
import pickle as pkl
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint


def tokenize(text):
    punctuation_map = str.maketrans('', '', string.punctuation)
    stopwords_list = stopwords.words('english')
    stopwords_list.remove('i')
    stopwords_list.remove('me')
    stopwords_list.append('com')
    stopwords_set = set(stopwords_list)
    text = text.split()
    # remove website links in text
    text = [word for word in text if not ('http' in word or 'www' in word)]
    text = [word.translate(punctuation_map).lower() for word in text]
    tokenized_words = [word for word in text if word not in stopwords_set]
    return tokenized_words


def make_sequence(tokens, length=21):
    sequences = []
    for i in range(length, len(tokens)+1):
        seq = tokens[i-length:i]
        line = ' '.join(seq)
        sequences.append(line)
    return sequences


def generate_sentence(model, tokenizer, sequence_length,
                      starting_text, num_predicted_words):
    prediction = [starting_text]
    for _ in range(num_predicted_words):
        encoded_text = tokenizer.texts_to_sequences([starting_text])[0]
        encoded_text = pad_sequences([encoded_text],
                                     maxlen=sequence_length,
                                     truncating='pre')
        preds = model.predict_classes(encoded_text, verbose=0)
        out_word = ''
        for word, idx in tokenizer.word_index.items():
            if idx == preds:
                out_word = word
                break
        starting_text += ' ' + out_word
        prediction.append(out_word)
    return ' '.join(prediction)


def make_tokenizer(token_sequences):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(token_sequences)
    num_sequences = tokenizer.texts_to_sequences(token_sequences)
    sequence_array = np.array(pad_sequences(num_sequences, padding='pre'))
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, sequence_array, vocab_size


def make_model(vocab_size, seq_length):
    model = Sequential()
    model.add(Embedding(vocab_size, seq_length, input_length=seq_length))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def train_model(model, X, y, path, batch_size=128, num_epochs=100):
    checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    model.fit(X, y, batch_size=batch_size, epochs=num_epochs,
              verbose=1, callbacks=[checkpoint])
    return model


if __name__ == '__main__':
    data = []
    with open('./data/recipeInfo.txt', 'r') as f:
        line = f.readline()
        while line:
            data.append(line[:-2])
            line = f.readline()
        f.close()

    df_recipe = pd.DataFrame(data, columns=['Recipe'])
    df_recipe['Length'] = df_recipe['Recipe'].apply(lambda x: len(x.split()))
    df_recipe['Unique Words'] = df_recipe['Recipe'].apply(lambda x: len(set(x.split())))
    df_recipe['Tokenized Recipe'] = df_recipe['Recipe'].apply(tokenize)

    seq_length = 20
    recipes_token = df_recipe['Tokenized Recipe'].tolist()
    token_sequences = [token for recipe in recipes_token for
                       token in make_sequence(recipe, length = seq_length+1)]
    tokenizer, sequence_array, vocab_size = make_tokenizer(token_sequences)

    X, y = sequence_array[:, :-1], sequence_array[:, -1]
    y = to_categorical(y, num_classes=vocab_size)
    model = make_model(vocab_size, seq_length)

    path = './model/checkpoints/recipe_model_ckpt.h5'
    trained_model = train_model(model, X, y, path)
    trained_model.save('./model/recipe_model_epoch100.h5')

    with open('./model/tokenizer.pkl', 'wb') as t:
        pkl.dump(tokenizer, t)




