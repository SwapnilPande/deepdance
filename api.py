import sys
from flask import Flask, request 
import os 

app = Flask(__name__)

os.environ["FLASK_ENV"] = "development"


@app.route("/test", methods=["POST"])
def create(): 
	request.files['teacher'].save("./videos/master.webm")
	request.files['student'].save("./videos/student.webm")

	#Get scores
	return "test" #return scores 

if __name__ == "__main__":
	app.run(debug=True, host='127.0.0.1', port=5000)