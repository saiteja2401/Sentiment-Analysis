import nltk
import pandas as pd
from nltk.corpus import stopwords
import joblib
from flask import Flask, request
from flask.templating import render_template

#flask initilaization
app = Flask(__name__)

#Homepage 
@app.route('/')
def home_page():
    return render_template('home_page.html')


@app.route('/', methods = ['get', 'post'])
def result():
    vector = joblib.load('tfidf_vector_model.pkl')
    model = joblib.load('netflix_75.pkl')
    input = request.form.get('paragraph')
    td = vector.transform(pd.Series(input))
    result = model.predict(td)
    f_result =""
    if result[0] == 1:
        f_result = 'A Positive Phrase'
    else:
        f_result = 'A Negative Phrase'
    return render_template('home_page.html', f_result = f_result, input = input)
if __name__ == '__main__':
    app.run(debug = True)
