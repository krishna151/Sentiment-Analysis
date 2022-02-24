from flask import Flask,render_template,request
import sqlite3 as sql
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import re
import pickle
from sklearn import metrics
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
nltk.download('stopwords')

forest = pickle.load(open('forest.pkl', 'rb'))
vectorize = pickle.load(open('vec.pkl', 'rb'))

app = Flask('__name__')

# Create a database
import sqlite3

conn = sqlite3.connect('movie_review.db')
print ("Opened database successfully")
conn.execute('DROP TABLE IF EXISTS movie')
conn.execute('CREATE TABLE movie (movie_review TEXT , prediction TEXT)')
print ("Table created successfully")
conn.close()

def review_cleaner(review):
    '''
    Clean and preprocess a review.
    
    1. Remove HTML tags
    2. Use regex to remove all special characters (only keep letters)
    3. Make strings to lower case and tokenize / word split reviews
    4. Remove English stopwords
    5. Rejoin to one string
    '''
    
    #1. Remove HTML tags
    review = BeautifulSoup(review).text
    
    #2. Use regex to find emoticons
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', review)
    
    #3. Remove punctuation
    review = re.sub("[^a-zA-Z]", " ",review)
    
    #4. Tokenize into words (all lower case)
    review = review.lower().split()
    
    #5. Remove stopwords
    eng_stopwords = set(stopwords.words("english"))
    review = [w for w in review if not w in eng_stopwords]
    
    #6. Join the review to one sentence
    review = ' '.join(review+emoticons)
    # add emoticons to the end

    return(review)

@app.route('/')
def index():
    return render_template("homepage.html")

@app.route('/home', methods=["GET","POST"]) # decorator to tell Flask what URL should trigger the function below
def home():
    if request.method == 'POST':
        a = request.form['fname']
    return render_template("data.html")

@app.route('/result', methods=["POST"]) # decorator to tell Flask what URL should trigger the function below
def result():
    a = request.form['fname']
    data = vectorize.transform([review_cleaner(a)])
    data = data.toarray()
    predict = forest.predict(data)
    out = predict[0]

    if out == 0:
        out = 'Negative 😔'
    else: 
        out = 'Positive 😀'
    with sql.connect("movie_review.db") as con:
        cur = con.cursor()
    
        cur.execute("INSERT INTO movie (movie_review,prediction) VALUES (?,?)",(a,out) ) # ? and tuple for placeholders
    
        con.commit()
    return render_template("result.html",out = out)

@app.route('/data', methods=["GET","POST"]) # decorator to tell Flask what URL should trigger the function below
def data():
    con = sql.connect("movie_review.db")
    con.row_factory = sql.Row
    cur = con.cursor()
    cur.execute("select * from movie")
    rows = cur.fetchall() # returns list of dictionaries
    return render_template("data.html",rows = rows)

if __name__ == '__main__':
    app.run(debug=True)