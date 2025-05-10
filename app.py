from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import gdown

# Initialize Flask app
app = Flask(__name__)

def download_from_gdrive(file_id, dest):
    if not os.path.exists(dest):
        print(f"Downloading {dest} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest, quiet=False)
    else:
        print(f"{dest} already exists. Skipping download.")

def download_all_models():
    files_to_download = {
        "accident_prediction_model.pkl": "1b6dWIeUduaVJLV8cswnQ1LkdE93rDze7",
        "imputer.pkl": "1I29lIxffoCA9_9Nuroide9l5WmrA9Mpm",
        "input_columns.pkl": "18lp7B5sZT88fbMTmQh1C65iFaml2uQO_",
        "logistic_regression_violation_model.pkl": "1xwXjT6AYgmAXTIX-PVYP8wXq-B20y-SL",
        "model1.pkl": "1kGDI9u8RR_14iI4mOGiXCVwaREUi03VM",
        "scaler.pkl": "1K1ySVMgYxbMPoNzIteONByST6LGXy_GD",
        "location_summary.csv": "17LO74gLpDDYCI9gXT-5ifB66KJQMG5Ok"
    }

    for filename, file_id in files_to_download.items():
        download_from_gdrive(file_id, filename)

download_all_models()


# Load all models and preprocessing tools
model = joblib.load("model1.pkl")  # Location-based risk model
model1 = joblib.load("logistic_regression_violation_model.pkl")  # Violation prediction model
model2 = joblib.load("accident_prediction_model.pkl")  # accident risk model

# Load additional required objects
df = pd.read_csv("location_summary.csv")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
input_columns = joblib.load("input_columns.pkl")

# map weekday names to numeric codes
day_map = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

@app.route("/")
def home():
    return render_template("index.html")


# Route:  High-Risk Locations model

@app.route("/day_and_time", methods=["GET", "POST"])
def day_and_time():
    results = None
    if request.method == "POST":
        day = request.form["day"]
        hour = int(request.form["hour"])
        month = int(request.form["month"])

        # Prepare data
        location_df = df[['location', 'location_cluster', 'latitude', 'longitude']].copy()
        location_df['day_of_week'] = day_map[day]
        location_df['hour'] = hour
        location_df['month'] = month

        input_features = ['location_cluster', 'longitude', 'latitude', 'month']
        X_input = location_df[input_features]

        # Predict risk
        location_df['risk_probability'] = model.predict_proba(X_input)[:, 1]

        # Sort by risk
        location_df = location_df.sort_values(by='risk_probability', ascending=False)

        # Round lat/lon to group nearby points 
        location_df['lat_rounded'] = location_df['latitude'].round(3)
        location_df['lon_rounded'] = location_df['longitude'].round(3)

        # Drop near-duplicate locations
        location_df = location_df.drop_duplicates(subset=['lat_rounded', 'lon_rounded'])

        # Take top 10
        results = location_df.head(10)[['location', 'latitude', 'longitude', 'risk_probability']].round(3).values.tolist()



    return render_template("based_on_day_and_time.html", results=results)




# Route: Violation Prediction model

@app.route("/specific_locations", methods=["GET", "POST"])
def specific_locations():
    results = None
    if request.method == "POST":
        latitude = float(request.form["latitude"])
        longitude = float(request.form["longitude"])
        user_input = [latitude, longitude]

        result = []
        for violation, m in model1.items():
            prob = m.predict_proba([user_input])[0][1]
            result.append((violation, prob))

        # Top 4 violations
        top_violations = sorted(result, key=lambda x: x[1], reverse=True)[:4]
        results = [(v, round(p, 3)) for v, p in top_violations]

    return render_template("specific_locations.html", results=results)

# Accident Risk Estimator

def predict_accident_risk(input_dict):
    df_input = pd.DataFrame([input_dict])
    df_input_encoded = pd.get_dummies(df_input)
    df_input_encoded = df_input_encoded.reindex(columns=input_columns, fill_value=0)
    df_imputed = pd.DataFrame(imputer.transform(df_input_encoded), columns=input_columns)
    df_scaled = scaler.transform(df_imputed)
    prob = model2.predict_proba(df_scaled)[0][1]
    return round(prob * 100, 2)

def simplified_accident_risk(user_input):
    base = {
        'hour': 12,
        'month': 6,
        'day_of_week': 'Monday',
        'gender': 'Male',
        'race': 'Unknown',
        'driver_state': 'MD',
        'vehicle_type': 'PASSENGER CAR',
        'arrest_type': 'Citation',
        'belts': True,
        'personal_injury': False,
        'property_damage': True,
        'fatal': False,
        'alcohol': False,
        'commercial_vehicle': False,
        'hazmat': False,
        'make': 12,
        'model': 103
    }
    base.update(user_input)
    return predict_accident_risk(base)


@app.route("/accidents_prediction", methods=["GET", "POST"])
def accidents_prediction():
    likelihood = None
    if request.method == "POST":
        hour = int(request.form["hour"])
        day_of_week = request.form["day"]
        vehicle_type = request.form["vehicle_type"]
        alcohol = request.form.get("alcohol") == "on"
        belts = request.form.get("belts") == "on"

        user_input = {
            "hour": hour,
            "day_of_week": day_of_week,
            "vehicle_type": vehicle_type,
            "alcohol": alcohol,
            "belts": belts
        }

        likelihood = simplified_accident_risk(user_input)

    return render_template("accidents_prediction.html", likelihood=likelihood)

# Run the app (optional for local testing)
if __name__ == "__main__":
    app.run(debug=True)
