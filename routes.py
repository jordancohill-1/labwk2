from flask import Flask, render_template
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
	prediction = model.predict([[4, 2.5, 3005, 15, 17903.0]]).round(2)
	prediction = np.squeeze(prediction)
	return render_template("index.html", prediction=prediction)

if __name__ == '__main__':
	model = joblib.load('regressor.pkl')
	app.run(debug=True)

	

	