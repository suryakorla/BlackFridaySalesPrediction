import numpy as np
from flask import Flask, request, render_template
import sklearn
import pickle
import pandas as pd
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
app = Flask(__name__)
model = pickle.load(open("black_rf.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        Gender=int(request.form["Gender"])
        Age = int(request.form["Age"])
        Occupation = int(request.form["Occupation"])
        city_category = int(request.form["City_Category"])
        Stay_In_Current_City_Years = int(request.form["Stay_In_Current_City_Years"])
        Marital_Status = int(request.form["Marital_Status"])
        Product_Category_1 = request.form["Product_Category_1"]
        Product_Category_2 = float(request.form["Product_Category_2"])


        prediction = model.predict([[Gender, Age,Occupation,city_category,Stay_In_Current_City_Years,Marital_Status,Product_Category_1,Product_Category_2]])
        x=np.exp(prediction)

        return render_template('home.html', prediction_text="Prediction Rs. {}".format(x))

    return render_template("home.html")

if __name__ == "__main__":
 app.run(debug=True)
