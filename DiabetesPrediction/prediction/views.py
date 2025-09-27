from django.shortcuts import render
import os
import joblib
from django.conf import settings

# Path to the model
model_path = os.path.join(settings.BASE_DIR, 'prediction', 'ml_models', 'diabetes_model.pkl')

# Load the model
model = joblib.load(model_path)
from django.shortcuts import render

def predict_diabetes(request):
    result = None
    if request.method == 'POST':
        # Example: collect input from form
        pregnancies = int(request.POST['pregnancies'])
        glucose = int(request.POST['glucose'])
        blood_pressure = int(request.POST['blood_pressure'])
        skin_thickness = int(request.POST['skin_thickness'])
        insulin = int(request.POST['insulin'])
        bmi = float(request.POST['bmi'])
        dpf = float(request.POST['dpf'])
        age = int(request.POST['age'])

        # Prepare input for model
        input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]

        # Predict
        prediction = model.predict(input_data)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

    return render(request, 'predict.html', {'result': result})
