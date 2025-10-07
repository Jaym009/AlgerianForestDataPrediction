import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application= Flask(__name__)
app = application

ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        data = [
            float(request.form.get('Temperature')),
            float(request.form.get('RH')),
            float(request.form.get('Ws')),
            float(request.form.get('Rain')),
            float(request.form.get('FFMC')),
            float(request.form.get('DMC')),
            float(request.form.get('ISI')),
            float(request.form.get('Classes')),
            float(request.form.get('Region'))
        ]
        final_data = standard_scaler.transform(np.array(data).reshape(1, -1))
        output = ridge_model.predict(final_data)[0]

        return render_template('home.html', result=round(output, 2))
    else:
        return render_template('home.html')


if __name__=='__main__':
    app.run(host='0.0.0.0')
