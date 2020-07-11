# From Python
# It requires OpenCV installed for Python
import sys
import cv2
from tqdm import tqdm
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

    def process_image(self, image):
        datum = op.Datum()
        datum.cvInputData = image
        self.opWrapper.emplaceAndPop([datum])
        return datum


    def process_image_path(self, path):
        imageToProcess = cv2.imread(path)
        return self.process_image(imageToProcess)

    def display_pose(self, datum):
        cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
        cv2.waitKey(0)


    def process_image_pair(self, image1, image2):
        '''
        Generates pose estimation results and evaluates them using DanceScorer
        :param image1: path to image
        :param image2: path to image
        :return:
        '''
        # Process and display images
        datum1 = op.Datum()
        datum1.cvInputData = image1
        self.opWrapper.emplaceAndPop([datum1])
        datum2 = op.Datum()
        datum2.cvInputData = image2
        self.opWrapper.emplaceAndPop([datum2])
        self.dance_scorer.add_frame_pose(datum1.poseKeypoints, datum2.poseKeypoints)
        return datum1, datum2

    def dance_end(self):
        # return self.dance_scorer.score_dancer()
        # FEEDBACK AND DISPLAY
        return 0

    def iterate_over_video(self, path):
        video = cv2.VideoCapture(path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_width = video.get(3)
        frame_height = video.get(4)
        frames = []
        with tqdm(total=video.get(cv2.CAP_PROP_FRAME_COUNT), desc='Processing') as pbar:
            while(1):
                success, frame = video.read()
                if success:
                    frames.append(self.process_image(frame).cvOutputData)
                    pbar.update(1)
                else:
                    break
        print('1/1')
        self.write_video('result.mp4', frames, fps, (int(frame_width), int(frame_height)))

    def compare_videos(self, path1, path2):
        video1 = cv2.VideoCapture(path1)
        fps1 = video1.get(cv2.CAP_PROP_FPS)
        frame_width1 = video1.get(3)
        frame_height1 = video1.get(4)

        video2 = cv2.VideoCapture(path2)
        fps2 = video2.get(cv2.CAP_PROP_FPS)
        frame_width2 = video2.get(3)
        frame_height2 = video2.get(4)

        frames1, frames2 = [],[]
        print('Processing')
        with tqdm(total=video1.get(cv2.CAP_PROP_FRAME_COUNT)) as pbar:
            while 1:
                success1, frame1 = video1.read()
                success2, frame2 = video2.read()
                if success1 and success2:
                    datum1, datum2 = self.process_image_pair(frame1, frame2)
                    frames1.append(datum1.cvOutputData)
                    frames2.append(datum2.cvOutputData)
                    pbar.update(1)
                else:
                    break
        print('1/2')
        self.write_video('video1Processed.mp4', frames1, fps1, (int(frame_width1), int(frame_height1)))
        print('2/2')
        self.write_video('video2Processed.mp4', frames2, fps2, (int(frame_width2), int(frame_height2)))
        return self.dance_end()

    def write_video(self, fname, frames,fps, shape):
        api = cv2.CAP_FFMPEG
        code = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
        output = cv2.VideoWriter(fname, api, code, fps, shape)
        with tqdm(total=len(frames), desc='Writing') as pbar:
            for frame in frames:
                output.write(frame)
                pbar.update(1)
        output.release()

if __name__ == "__main__":

    pose_estimator = PoseEstimator()
    # # Process Image
    # datum = pose_estimator.process_image_path('COCO_val2014_000000000192.jpg')
    # # Display Image
    # pose_estimator.display_pose(datum)

    fname1 = 'videos/macarena.mp4'
    fname2 = 'videos/ymca.mp4'
    # print(pose_estimator.compare_videos(fname1, fname2))
    pose_estimator.iterate_over_video('videos/macarena.mp4')