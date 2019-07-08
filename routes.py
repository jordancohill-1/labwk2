from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)
Bootstrap(app)

@app.route("/")
def index():
	model = joblib.load('regressor.pkl')
	prediction = model.predict([[4, 2.5, 3005, 15, 17903.0]]).round(1)
	prediction = np.squeeze(prediction)
	return render_template("index.html", prediction=prediction)


if __name__ == '__main__':
	app.run(debug=True)

	

	