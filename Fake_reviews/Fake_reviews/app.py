from flask import Flask, render_template, request, redirect, url_for, session
# from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pickle as pkl
import os
import re 
import ftfy
import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
# import spacy
# from spacy.lang.en.stop_words import STOP_WORDS
# from spacy.lang.en import English
import string
import warnings
warnings.filterwarnings("ignore")
# parser = English()
punctuations = string.punctuation
#
# stopwords = set(STOP_WORDS)

app = Flask(__name__)
app.secret_key = 'key'

class_names = ['Negative', 'Positive']


# nlp = spacy.load("en_core_web_sm")

cList = pkl.load(open('model/cword_dict.pkl', 'rb'))
trained_tokenizer = pkl.load(open('model/tokens.pkl', 'rb'))
c_re = re.compile('(%s)' % '|'.join(cList.keys()))

trained_model = load_model("model/CNN-LSTM_model.h5")
names = ['user', 'product', 'review']


def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)


def clean_review(reviews):
    cleaned_review = []
    for review in reviews:
        review = str(review)
        if re.match("(\w+:\/\/\S+)", review) == None and len(review) > 10:
            review = ' '.join(re.sub("(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)", " ", review).split())
            review = ftfy.fix_text(review)
            review = expandContractions(review)
            review = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", review).split())
            stop_words = stopwords.words('english')
            word_tokens = nltk.word_tokenize(review)
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            review = ' '.join(filtered_sentence)
            review = PorterStemmer().stem(review)
            cleaned_review.append(review)
    return cleaned_review


@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        excel_file = request.files['file']
        fileName = "Dataset/" + "dataset.xls"
        excel_file.save(fileName)
        msg = "Validation File Uploaded Successfully"
        return render_template("home.html", msg=msg)


@app.route("/submit", methods=['GET', 'POST'])
def get_hours():
    df = pd.read_excel('Dataset/dataset.xls', names=names)
    df.dropna(axis=0, inplace=True)
    test_data = df['review']
    cleaned_text = clean_review(test_data)
    sequences_text_token = trained_tokenizer.texts_to_sequences(cleaned_text)
    print(sequences_text_token)
    data = pad_sequences(sequences_text_token, maxlen=200)
    print('Shape of data tensor:', data.shape)

    def predict_result(data, model):
        result = model.predict(data)
        Y_pred = np.round(result.flatten())

    Y_pred = predict_result(data=data, model=trained_model)
    j = trained_model.predict(data)
    rec = len(test_data) - len(j)
    k = np.round(j.flatten())
    k = list(k)
    for i in range(rec):
        k.append(0)
    df['Predict'] = k
    df['Predict'] = df.Predict.astype(int)
    df.user.value_counts()
    df.to_excel('Dataset/sdata.xlsx', index=False)
    predict_value = df['Predict'].tolist()
    prodect_value = df.iloc[:, 1].values.tolist()
    pro = df.iloc[:, 1].unique()
    my_dict = {}
    for i in pro:
        try:
            i = i.strip()
            my_dict[i] = []
        except:
            my_dict[i] = []

    for i1, j1 in df.iterrows():
        for i2, j2 in my_dict.items():
            if j1[1] == i2:
                if j1['Predict'] == 0:
                    my_dict[i2].append(0)
                else:
                    my_dict[i2].append(1)
    prodect_senti = []
    for i, j in my_dict.items():

        pos = j.count(0)
        neg = j.count(1)
        if pos >= neg:
            prodect_senti.append('Pos')
        else:
            prodect_senti.append('Neg')

    data = {'Prodect': pro, 'Review': prodect_senti}
    pf = pd.DataFrame(data)
    user = df.user.unique()
    my_user = {}
    for i in user:
        my_user[i] = []

    for ind, row in df.iterrows():
        for ind1, row1 in pf.iterrows():
            if row[1] == row1['Prodect']:
                n = ''
                if row['Predict'] == 0:
                    n = 'Pos'
                else:
                    n = 'Neg'
                if n == row1['Review']:
                    for n1, m in my_user.items():
                        if row['user'] == n1:
                            my_user[n1].append('Yes')
                else:
                    for n1, m in my_user.items():
                        if row['user'] == n1:
                            my_user[n1].append('No')
    with_crowd = []
    with_out_crowd = []
    for i, j in my_user.items():
        with_crowd.append(j.count('Yes'))
        with_out_crowd.append(j.count('No'))

    data = {'user': user, 'with_crowd': with_crowd, 'with_out_crowd': with_out_crowd}
    new_df = pd.DataFrame(data)

    new_df1 = new_df.copy()
    new_df1["with_crowd"] = new_df["with_crowd"] * 2
    new_df1["with_out_crowd"] = new_df["with_out_crowd"] * 2
    new_df["score"] = new_df1["with_crowd"] - new_df1["with_out_crowd"]
    new_df.sort_values(by="score", ascending=False).head(10)

    # new_df.sort_values("with_crowd", axis=0, ascending=False,inplace=True, na_position='last')
    # print(new_df.head(10))

    cols = list(new_df.columns)
    df = np.asarray(new_df)
    data1 = df
    return render_template("result.html", cols=cols, df=data1)


@app.route('/')
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form["email"]
        pwd = request.form["password"]
        r1 = pd.read_excel("user.xlsx")
        for index, row in r1.iterrows():
            if row["email"] == str(email) and row["password"] == str(pwd):
                return redirect(url_for('home'))
        else:
            msg = 'Invalid User, Please Try Again!'
            return render_template('login.html', msg=msg)
    return render_template('login.html')


@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['Email']
        Password = request.form['Password']
        col_list = ["name", "email", "password"]
        r1 = pd.read_excel('user.xlsx', usecols=col_list)
        new_row = {'name': name, 'email': email, 'password': Password}
        r1 = r1.append(new_row, ignore_index=True)
        r1.to_excel('user.xlsx', index=False)
        print("Records created successfully")
        # msg = 'Entered Mail ID Already Existed'
        msg = 'Registration Successful !! U Can login Here !!!'
        return render_template('login.html', msg=msg)
    return render_template('login.html')


@app.route("/home", methods=['GET', 'POST'])
def home():
    return render_template("home.html")


@app.route('/password', methods=['POST', 'GET'])
def password():
    if request.method == 'POST':
        current_pass = request.form['current']
        new_pass = request.form['new']
        verify_pass = request.form['verify']
        r1 = pd.read_excel('user.xlsx')
        for index, row in r1.iterrows():
            if row["password"] == str(current_pass):
                if new_pass == verify_pass:
                    r1.replace(to_replace=current_pass, value=verify_pass, inplace=True)
                    r1.to_excel("user.xlsx", index=False)
                    msg1 = 'Password changed successfully'
                    return render_template('password_change.html', msg1=msg1)
                else:
                    msg2 = 'Re-entered password is not matched'
                    return render_template('password_change.html', msg2=msg2)
        else:
            msg3 = 'Incorrect password'
            return render_template('password_change.html', msg3=msg3)
    return render_template('password_change.html')


@app.route('/graphs', methods=['POST', 'GET'])
def graphs():
    return render_template('Graphs.html')


@app.route('/knn')
def knn():
    return render_template('knn.html')


@app.route('/logout')
def logout():
    session.clear()
    msg = 'You are now logged out', 'success'
    return redirect(url_for('login', msg=msg))


if __name__ == '__main__':
    app.run(port=3000, debug=True)



