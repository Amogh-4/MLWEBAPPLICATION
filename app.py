from flask import Flask, render_template, request
import pickle
import numpy as np

f = open('iris (1).pkl', 'rb')
model = pickle.load(f)
f.close()

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def home():
    SL = request.form['a']
    SW = request.form['b']
    PL = request.form['c']
    PW = request.form['d']
    arr = np.array([[float(SL), float(SW), float(PL), float(PW)]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)

if __name__ == '__main__':
    app.run(debug = True)
