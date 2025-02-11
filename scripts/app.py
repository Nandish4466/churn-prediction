from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get values from form
        features = [float(request.form[f]) for f in request.form]
        prediction = model.predict([features])[0]
        result = "Churn" if prediction == 1 else "No Churn"

        return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
