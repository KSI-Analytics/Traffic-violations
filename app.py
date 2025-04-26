from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load model and data
model = joblib.load("model.pkl")
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

if __name__ == "__main__":
    app.run(debug=True)