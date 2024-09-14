from flask import Flask, render_template , request
import pickle

cv = pickle.load(open("models/cv.pkl","rb"))
clf = pickle.load(open("models/clf.pkl","rb"))
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["post"])
def predict():
    email_text = ""
    if request.method == 'POST':
        email_text = request.form.get('email-content')
    tokenized_email = cv.transform([email_text])
    predict_output = clf.predict(tokenized_email)
    predictions = 1 if predict_output == 1 else 0 
    return render_template("index.html",email_text = email_text, predictions = predictions)

if __name__ == "__main__":
    app.run(debug=True)
    # app.run()