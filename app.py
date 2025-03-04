from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("churn_model_Bank.pkl", "rb"))

# Load the customer dataset
df = pd.read_csv("cleaned_data_Bank.csv")

@app.route("/")
def home():
    return "Customer Churn Prediction API"

@app.route("/predict", methods=["POST"])
def predict():
    # Get the features from the POST request
    data = request.json["features"]
    
    # Make a prediction using the model
    prediction = model.predict([np.array(data)])
    
    # Return the result as JSON
    return jsonify({"Churn Prediction": int(prediction[0])})

@app.route("/index")
def index():
    # Render the HTML template with the customer dataset passed as a table
    return render_template("index.html", data=df.to_html(classes='table table-striped'))

if __name__ == "__main__":
    app.run(debug=True)
