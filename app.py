from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved SVM model from the file
with open('svm_model.pkl', 'rb') as file:
    model = pickle.load(file)


# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the HTML form
    gender = request.form['gender']
    married = request.form['married']
    education = request.form['education']
    self_employed = request.form['self_employed']
    dependents = float(request.form['dependents'])
    applicant_income = float(request.form['applicant_income'])
    loan_amount = float(request.form['loan_amount'])
    loan_amount_term = float(request.form['loan_amount_term'])

    labelencoder_X = LabelEncoder()

    # Transform user input using the label encoder
    gender = labelencoder_X.fit_transform([gender])[0]
    married = labelencoder_X.fit_transform([married])[0]
    education = labelencoder_X.fit_transform([education])[0]
    self_employed = labelencoder_X.fit_transform([self_employed])[0]

    # Create input array for prediction
    user_data = np.array([[gender, married, dependents, education, self_employed, applicant_income, loan_amount, loan_amount_term]])

    # Make prediction using the SVM model
    prediction = model.predict(user_data)

    # Convert prediction label to human-readable form
    if prediction == 1:
        eligibility = "Eligible"
    else:
        eligibility = "Not Eligible"

    # Render the prediction result on the result page
    return render_template('index.html', prediction=eligibility)

if __name__ == '__main__':
    app.run(debug=True)
