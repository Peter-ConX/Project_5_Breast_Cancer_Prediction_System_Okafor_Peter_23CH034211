from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open("model/breast_cancer_model.pkl", "rb") as file:
    model, scaler = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        radius = float(request.form["radius"])
        texture = float(request.form["texture"])
        perimeter = float(request.form["perimeter"])
        area = float(request.form["area"])
        smoothness = float(request.form["smoothness"])

        data = np.array([[radius, texture, perimeter, area, smoothness]])
        data = scaler.transform(data)

        result = model.predict(data)[0]
        prediction = "Benign" if result == 1 else "Malignant"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
