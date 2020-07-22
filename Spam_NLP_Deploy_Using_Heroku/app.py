import pickle

from flask import Flask, render_template, request

# load the model from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('tranform.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
	app.run(port=2000, debug=True)
