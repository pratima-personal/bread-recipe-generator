#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
import pickle as pkl
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')


def generate_sentence(model, tokenizer, sequence_length, starting_text, num_predicted_words):
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


if __name__ == '__main__':
    trained_model = load_model('model/recipe_model_epoch100.h5')
    with open('model/tokenizer_model.pkl', 'rb') as t:
        tokenizer = pkl.load(t)

    starting_text = sys.argv[1]
    seq_length = 20
    pred_length = 50
    predicted_text = generate_sentence(trained_model, tokenizer,
                                       seq_length, starting_text, pred_length)
    print(f'\nThe next {pred_length} words predicted for {starting_text} are:')
    print(predicted_text)
    
    

