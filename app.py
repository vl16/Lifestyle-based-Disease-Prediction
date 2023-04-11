from flask import Flask,render_template,url_for,request
from flask_material import Material
from sklearn.ensemble import RandomForestClassifier

# EDA PKg
import pandas as pd 
import numpy as np 

# ML Pkg
import joblib
import os
import pickle

app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview')
def preview():
    df = pd.read_csv("data/d_sih1.csv")
    return render_template("preview.html",df_view = df)

@app.route('/',methods=["POST"])
def analyze():
	if request.method == 'POST':
		
		age_input = request.form['age_input']
		# height_input = request.form['height_input'] 

		# bmi = (weight_input/(height_input/100)**2)

		gender_choice = request.form['gender_choice']
		smoking_input = request.form['smoking_input']
		exercise_input = request.form['exercise_input']
		drinking_input = request.form['drinking_input']
		bmi_input = request.form['bmi_input']
		# idhr se input le rhe hai html se
		sleep_input = request.form['sleep_input']
		# model_choice = request.form['model_choice']
		# weight_input = request.form['weight_input'] 
		junk_input = request.form['junk_input']

		# h = float(height_input)
		# w = float(weight_input)
		#bmi = (w/(h/100)**2)
		age = float(age_input)
		sex = float(gender_choice)
		bmi= float(bmi_input)
		smoking = float(smoking_input)
		excercise = float(exercise_input)
		sleep = float(sleep_input)
		drinking = float(drinking_input)
		junk = float(junk_input)
		
		# Clean the data by convert from unicode to float 
		sample_data = [age_input,bmi_input,drinking_input,exercise_input,gender_choice,junk_input,sleep_input,smoking_input]
		clean_data = [float(i) for i in sample_data]
		# lean_data = [int(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)

		# reloading model 
		logit_model = pickle.load(open('life.pkl', 'rb'))
		result_prediction = logit_model.predict(ex1)
		result_prediction = int(result_prediction)

	return render_template('index.html', 
		
		age_input = age_input,
		# height_input = height_input,
		gender_choice = gender_choice,
		# weight_input = weight_input,
		sleep_input = sleep_input,
		junk_input = junk_input,
		smoking_input = smoking_input,
		exercise_input = exercise_input,
		drinking_input = drinking_input,
		bmi_input = bmi_input,

		clean_data=clean_data,
		result_prediction=result_prediction)
		# model_selected=model_choice)


# Run app
if __name__ == '__main__':
    app.run(debug=True) 