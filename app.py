from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model_laptop.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract input data from the form
            inches = float(request.form['inches'])
            ram = int(request.form['ram'])
            weight = float(request.form['weight'])
            
            # Prepare data for prediction
            input_features = np.array([[inches, ram, weight]])
            input_features_scaled = scaler.transform(input_features)
            
            # Predict using the model
            prediction = model.predict(input_features_scaled)[0]
        
        except Exception as e:
            return str(e)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
