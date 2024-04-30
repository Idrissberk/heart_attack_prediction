import joblib
import pandas as pd
from flask import Flask, request

app = Flask(__name__)

@app.get("/heart_attack_risk")
def api_predict_heart_attact():
    regr = joblib.load("model.pkl")

    age = int(request.args.get("age", 0))
    cholesterol = int(request.args.get("cholesterol", 0))
    blood_pressure = request.args.get("blood_pressure", "120/80").split("/")
    smoking = request.args.get("smoking", "0") == "1"
    alcohol = request.args.get("alcohol", "0") == "1"
    diabetes = request.args.get("diabetes", "0") == "1"
    obesity = request.args.get("obesity", "0") == "1"
    previous = int(request.args.get("previous", 0))
    heart_rate = int(request.args.get("heart_rate", 75))
    sex = request.args.get("sex", "")

    return {
        "result": regr.predict(pd.DataFrame(data = {
          "Age": [age],
          "Cholesterol": [cholesterol],
          "Smoking": int(smoking),
          "Alcohol Consumption": int(alcohol),
          "Diabetes": int(diabetes),
          "Obesity": int(obesity),
          "Previous Heart Problems": previous,
          "Heart Rate": heart_rate,
          "Blood Pressure (systolic)": [float(blood_pressure[0])],
          "Blood Pressure (diastolic)": [float(blood_pressure[1])],
          "Sex_Male": [int(sex == "Male")],
          "Sex_Female": [int(sex == "Female")]
      })).astype(float)[0]
    }

