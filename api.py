import sys
from flask import Flask, request 
import os
import pose_estimation
import json
import time

app = Flask(__name__)

os.environ["FLASK_ENV"] = "development"


@app.route("/test", methods=["POST"])
def create(): 
	request.files['teacher'].save("/home/david/Documents/DeepDance/deepdance/videos/master.mp4")
	request.files['student'].save("/home/david/Documents/DeepDance/deepdance/videos/student.mp4")

	pose_estimator = pose_estimation.PoseEstimator()
	result = json.dumps(pose_estimator.compare_videos("/home/david/Documents/DeepDance/deepdance/videos/student.mp4",
													"/home/david/Documents/DeepDance/deepdance/videos/master.mp4",
													write_skeleton=True,
													  skeleton_out1="/home/david/Documents/DeepDance/deepdance/videos/student-overlay.mp4",
													  skeleton_out2="/home/david/Documents/DeepDance/deepdance/videos/master-overlay.mp4",
                       write_aligned=False, aligned_out1='videos/aligned-david-choreo.mp4', aligned_out2='videos/aligned-davidcaro-choreo.mp4',
                       write_combined=False, combined_out='verification/david-caro-choreo.mp4'))
	pose_estimator.get_wireframe("/home/david/Documents/DeepDance/deepdance/videos/black-wireframe.mp4")
	return result # return scores

if __name__ == "__main__":
	app.run(debug=False, host='127.0.0.1', port=5000)