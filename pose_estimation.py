# From Python
# It requires OpenCV installed for Python
import sys
import cv2
from DanceScorer import DanceScorer

try:
    sys.path.append('/usr/local/python')
    from openpose import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

class PoseEstimator:
    def __init__(self):
        # parameters for pose estimation
        self.params = dict()
        self.params["model_folder"] = "models/"
        self.params["face"] = False
        self.params["hand"] = False
        # self.params["num_gpu"] = op.get_gpu_number()

        # Starting OpenPose
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(self.params)
        self.opWrapper.start()

        self.dance_scorer = DanceScorer()

    def process_image(self, path):
        datum = op.Datum()
        imageToProcess = cv2.imread(path)
        datum.cvInputData = imageToProcess
        self.opWrapper.emplaceAndPop([datum])
        return datum

    def display_pose(self, datum):
        cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
        cv2.waitKey(0)


    def process_image_pair(self, image1, image2):
        '''
        Generates pose estimation results and evaluates them using DanceScorer
        :param image1: cv2 image
        :param image2: cv2 image
        :return:
        '''
        # Process and display images
        datum1 = op.Datum()
        imageToProcess = image1
        datum1.cvInputData = imageToProcess
        self.opWrapper.emplaceAndPop([datum1])
        datum2 = op.Datum()
        imageToProcess = image2
        datum1.cvInputData = imageToProcess
        self.opWrapper.emplaceAndPop([datum2])
        return self.dance_scorer.add_frame_pose(datum1.poseKeypoints, datum2.poseKeypoints)

if __name__ == "__main__":

    pose_estimator = PoseEstimator()
    # Process Image
    datum = pose_estimator.process_image('COCO_val2014_000000000192.jpg')
    # Display Image
    pose_estimator.display_pose(datum)
