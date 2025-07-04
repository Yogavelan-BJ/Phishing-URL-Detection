#importing required libraries

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import joblib
warnings.filterwarnings('ignore')
from feature import FeatureExtraction

model = joblib.load("models/phishing_model.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, -1) 

        y_pred = model.predict(x)[0]
        #1 is safe       
        #-1 is unsafe
        y_pro_phishing = model.predict_proba(x)[0,0]
        y_pro_non_phishing = model.predict_proba(x)[0,1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        return render_template('index.html',xx =round(y_pro_non_phishing,2),url=url )
    return render_template("index.html", xx =-1)


if __name__ == "__main__":
    app.run(debug=True)