from flask import Flask, flash, redirect, render_template, request, url_for
import re
import base64
import model as mod
import os
import numpy as np
import keras


app = Flask(__name__)


my_model = mod.get_mnist_model()


# decoding an image from base64 into raw representation
def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))


@app.route('/s')
def index():
    return render_template('new_home.html', data=[{'name': 'Mnist'}, {'name': 'ImageNet'}])


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    # whenever the predict method is called, we're going
    # to input the user drawn character as an image into the model
    # perform inference, and return the classification
    # get the raw data format of the image
    imgData = request.get_data()
    convertImage(imgData)
    keras.backend.clear_session()
    my_model = mod.get_mnist_model()
    res = mod.get_result('output.png', my_model)
    print(np.round(res*100,2)) # probabilites of all other digits
    return str(np.argmax(res))



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5005)
