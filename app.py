from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

try:
    with open("history.pkl", "rb") as file:
        history = pickle.load(file)
    if not isinstance(history, list):  
        history = []  
except (FileNotFoundError, pickle.UnpicklingError):  
    history = []

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_price = None
    if request.method == "POST":
        try:
            sqft = float(request.form["sqft"])
            bedrooms = int(request.form["bedrooms"])
            input_features = np.array([[sqft, bedrooms]])
            predicted_price = model.predict(input_features)[0]
            predicted_price = round(float(predicted_price), 2)

            entry = {"sqft": sqft, "bedrooms": bedrooms, "price": predicted_price}
            history.append(entry)

            with open("history.pkl", "wb") as file:
                pickle.dump(history, file)

        except Exception as e:
            predicted_price = f"Error: {str(e)}"

    return render_template("index.html", predicted_price=predicted_price, history=history)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
