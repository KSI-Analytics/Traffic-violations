from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load model and data
model = joblib.load("model1.pkl")
df = pd.read_csv("cleaned_traffic_violations1.csv")

# Flask app
app = Flask(__name__)

# Day mapping
day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
           'Friday': 4, 'Saturday': 5, 'Sunday': 6}

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    if request.method == "POST":
        day = request.form["day"]
        hour = int(request.form["hour"])
        month = int(request.form["month"])

        # Prepare input
        location_df = df[['location', 'location_cluster', 'latitude', 'longitude']].drop_duplicates()
        location_df['day_of_week'] = day_map[day]
        location_df['hour'] = hour
        location_df['month'] = month

        input_features = ['location_cluster', 'longitude', 'latitude', 'month']
        X_input = location_df[input_features]

        location_df['risk_probability'] = model.predict_proba(X_input)[:, 1]
        results = location_df.sort_values(by='risk_probability', ascending=False).head(10)[
            ['location', 'latitude', 'longitude', 'risk_probability']
        ].round(3).values.tolist()

    return render_template("index.html", results=results)

model1 = joblib.load("logistic_regression_violation_model.pkl")

# First route: latitude + longitude input
@app.route("/specific_locations", methods=["GET", "POST"])
def specific_locations():
    results = None
    if request.method == "POST":
        latitude = float(request.form["latitude"])
        longitude = float(request.form["longitude"])

        user_input = [latitude, longitude]

        # Predict probabilities
        result = []
        for v in model1.keys():
            model = model1[v]
            p = model.predict_proba([user_input])[0][1]
            result.append((v, p))

        # Sort and select top 4
        top_violations = sorted(result, key=lambda x: x[1], reverse=True)[:4]

        results = [(vio, round(prob, 3)) for vio, prob in top_violations]

    return render_template("specific_locations.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)