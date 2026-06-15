from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    models = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model_names = ['Decision Tree', 'SVC']

@app.route('/')
def index():
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        pregnancies = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        blood_pressure = int(request.form['blood_pressure'])
        skin_thickness = int(request.form['skin_thickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        selected_model_name = request.form['model']
        model_idx = model_names.index(selected_model_name)
        model = models[model_idx]

        data = {
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age]
        }
        df = pd.DataFrame(data)

        df_scaled = scaler.transform(df)

        prediction_code = model.predict(df_scaled)[0]
        
        if prediction_code == 1:
            prediction = "Diabetic"
        else:
            prediction = "Non-Diabetic"

        return render_template(
            'index.html', 
            model_names=model_names, 
            prediction=prediction, 
            selected_model=selected_model_name
        )

if __name__ == '__main__':
    app.run(debug=True)