from flask import Flask,render_template,request
import joblib
import numpy as np

app = Flask(__name__)

MODEL_PATH = "artifacts/models/model.pkl"
SCALER_PATH = "artifacts/processed/scaler.pkl"

modal = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = [
                'Operation_Mode', 'Temperature_C', 'Vibration_Hz',
                'Power_Consumption_kW', 'Network_Latency_ms', 'Packet_Loss_%',
                'Quality_Control_Defect_Rate_%', 'Production_Speed_units_per_hr',
                'Predictive_Maintenance_Score', 'Error_Rate_%','year', 'month', 'day', 'hour'
            ]

LABLES= {
    0:"High",
    1:"Low",
    2:"Mediam"
}

@app.route('/',methods=['GET','POST'])
def index():
    prediction=None
    if request.method == 'POST':
        try:
            input_data = [float(request.form[feature]) for feature in FEATURES]
            input_array = np.array(input_data).reshape(1,-1)
            scaled_array = scaler.transform(input_array)

            pred = modal.predict(scaled_array)[0]
            prediction = LABLES.get(pred,"Unknown")
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html",prediction = prediction, features = FEATURES)

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=5000) 