from flask import Flask, jsonify
import pickle as pkl
import generate_recipe
from keras.models import load_model

trained_model = load_model('model/recipe_model_epoch100.h5')
with open('model/tokenizer_model.pkl', 'rb') as t:
    tokenizer = pkl.load(t)

app = Flask(__name__)


@app.route('/')
def hello():
    return 'hello world!'


@app.route('/predict/<text>')
def predict(text):
    seq_length = 20
    pred_length = 50
    predicted_text = generate_recipe.generate_sentence(trained_model, tokenizer,
                                                       seq_length, text, pred_length)
    return predicted_text


if __name__ == '__main__':
    app.run()


