import sys
from flask import Flask, request 
import os
import pose_estimation
import json

app = Flask(__name__)

os.environ["FLASK_ENV"] = "development"


@app.route("/test", methods=["POST"])
def create(): 
	request.files['teacher'].save("./videos/master.webm")
	request.files['student'].save("./videos/student.webm")

	pose_estimator = pose_estimation.PoseEstimator()

	#Get scores
	return json.dumps(pose_estimator.compare_videos("./videos/student.webm", "./videos/master.webm", write_skeleton=False, skeleton_out1='', skeleton_out2='',
                       write_aligned=False, aligned_out1='videos/aligned-david-choreo.mp4', aligned_out2='videos/aligned-davidcaro-choreo.mp4',
                       write_combined=False, combined_out='verification/david-caro-choreo.mp4')) #return scores

if __name__ == "__main__":
	app.run(debug=True, host='127.0.0.1', port=5000)