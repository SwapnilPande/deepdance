# From Python
# It requires OpenCV installed for Python
import sys
import cv2
from tqdm import tqdm
from DanceScorer import DanceScorer
from alignment import align, check_alignment

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
        self.params["number_people_max"] = 1
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
        assert datum1.poseKeypoints.shape == (1, 25, 3)
        assert datum2.poseKeypoints.shape == (1, 25, 3)
        self.dance_scorer.add_frame_pose(datum1.poseKeypoints, datum2.poseKeypoints)
        return datum1, datum2

    def dance_end(self):
        return self.dance_scorer.score_dancer()
        # FEEDBACK AND DISPLAY
        # return 0

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

    def compare_videos(self, path1, path2, write_skeleton=False, skeleton_out1='', skeleton_out2='',
                       write_aligned=False, aligned_out1='', aligned_out2='',
                       write_combined=False, combined_out=''):
        frames1, frames2, fps, shape1, shape2 = align(path1, path2, outpath1=aligned_out1,
                                                      outpath2=aligned_out2,
                                                      write=write_aligned)

        cvOut1 = []
        cvOut2 = []
        for i in tqdm(range(len(frames1))):
            datum1, datum2 = self.process_image_pair(frames1[i], frames2[i])
            cvOut1.append(datum1.cvOutputData)
            cvOut2.append(datum2.cvOutputData)
        if write_skeleton:
            print('1/2')
            self.write_video(skeleton_out1, cvOut1, fps, shape1)
            print('2/2')
            self.write_video(skeleton_out2, cvOut2, fps, shape2)
        if write_combined:
            check_alignment(frames1, frames2, fps, shape1, shape2, combined_out)
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

    def get_wireframe(self, outpath):
        self.dance_scorer.generate_wireframe_video(outpath)

if __name__ == "__main__":

    pose_estimator = PoseEstimator()

    fname1 = 'videos/david-ymca.mp4'
    fname2 = 'videos/ymca.mp4'
    print(pose_estimator.compare_videos(fname1, fname2, write_skeleton=False, skeleton_out1='', skeleton_out2='',
                       write_aligned=False, aligned_out1='videos/aligned-david-choreo.mp4', aligned_out2='videos/aligned-davidcaro-choreo.mp4',
                       write_combined=False, combined_out='verification/david-caro-choreo.mp4'))