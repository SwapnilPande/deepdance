import sys
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World from Flask!"


@app.route("/test", methods=["POST"])
def create(): 
	print(request.files['teacher'])
	print(request.files['student'])
	return //Return JSON

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)

