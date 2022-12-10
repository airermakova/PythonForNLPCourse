from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, current_app
from urllib.request import urlopen
import mechanicalsoup

import sys
import os

testmodelpath=os.getcwd()+"/testmodel"


app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')


def allowed_txt_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in "txt"


@app.route('/basic/', methods=['GET','POST'])
def basic():
    path=""
    output=""
    args = []
    if request.method == "POST":
        data = request.form['helloWord']
        print(data)                   
    return render_template('helloWord.html')


@app.route('/textproc/', methods=['GET','POST'])
def textproc():
    path=""
    output=""
    args = []
    if request.method == "POST":
        print("POST")            
    return render_template('index.html')


@app.route('/nn/', methods=['GET','POST'])
def nn():
    path=""
    output=""
    if request.method == "POST":
        print("POST ")
    return render_template('index.html')




if __name__ == '__main__':
    app.run(debug=True)


