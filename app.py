from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    study_hours = float(request.form['study_hours'])
    attendance = float(request.form['attendance'])
    assignment = float(request.form['assignment'])
    midterm = float(request.form['midterm'])
    final = float(request.form['final'])

    features = np.array([[study_hours,
                          attendance,
                          assignment,
                          midterm,
                          final]])

    prediction = model.predict(features)

    return render_template(
        'index.html',
        prediction_text=f"Predicted Grade: {prediction[0]}"
    )

if __name__ == "__main__":
    app.run(debug=True)