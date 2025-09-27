# DiabetesPrediction System
This repository contains a web-based Diabetes Prediction System that uses machine learning (Logistic Regression , Random Forest,KNN,SVM) trained on the Pima Indian Diabetes Dataset.
Users can enter their health parameters (e.g., Glucose, Insulin, BMI, Age, etc.), and the system predicts whether they are likely to have diabetes.


##Features

1.Machine learning–based prediction
2.Web-based user interface
3.Input form for health parameters
4.Instant prediction results
5.Result analysis and downloadable report


##How it Works

1. The user enters their health data (Glucose, Insulin, BMI, Age, etc.).
2. The system processes the data using a trained ML model (e.g.,KNN,SVM,Random Forest,Logistic Regression).
3. The prediction result is displayed, indicating whether the user is likely to have diabetes.

##Technologies Used

*Python – Machine Learning model (Logistic Regression)
*ML(scikit-learn,pandas,numpy)
*HTML, CSS, JavaScript – Frontend (user interface)
*Django – Backend (connecting frontend with ML model) (use whichever you actually implemented)


##Installation

1. Clone this repository
2. Install required libraries:
pip install -r requirements.txt
3. Run the backend server (Flask/Django):
python manage.py runserver
4. Open the project in your browser:
http://127.0.0.1:8000/




