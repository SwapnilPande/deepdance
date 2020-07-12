import numpy as np
from scipy.stats import norm

from enum import IntEnum

class Joint(IntEnum):
    NOSE = 0
    NECK = 1
    RSHOULDER = 2
    RELBOW = 3
    RWRIST = 4
    LSHOULDER = 5
    LELBOW = 6
    LWRIST = 7
    MIDHIP = 8
    RHIP = 9
    RKNEE = 10
    RANKLE = 11
    LHIP = 12
    LKNEE = 13
    LANKLE = 14
    REYE = 15
    LEYE = 16
    REAR = 17
    LEAR = 18
    LBIGTOE = 19
    LSMALLTOE = 20
    LHEEL = 21
    RBIGTOE = 22
    RSMALLTOE = 23
    RHEEL = 24



class DanceScorer:
    # Range values for the min-max joint angles
    RANGE = {
                "lshoulder" : 3.099920992,
                "rShoulder" : 3.115298634,
                "lelbow" : 3.139355155,
                "relbow" : 3.140255528,
                "lhip" : 2.306497931,
                "rhip" : 2.352498353,
                "lknee" : 2.422342539,
                "rknee" : 2.163526058,
                "lankle" : 2.097560167,
                "rankle" : 2.271983564
            }
    SIGMA_SCALE = 12

    def __init__(self):

        # Instantiate two lists to store the teacher and student poses
        self.poses = {
            "student" : [],
            "teacher" : []
        }


        # Each element in this dictionary is a list of length n storing the tracked metrics
        self.position_metrics = {
            "student" : {
                "lshoulder" : [],
                "rShoulder" : [],
                "lelbow" : [],
                "relbow" : [],
                "lhip" : [],
                "rhip" : [],
                "lknee" : [],
                "rknee" : [],
                "lankle" : [],
                "rankle" : []
            },
            "teacher" : {
                "lshoulder" : [],
                "rShoulder" : [],
                "lelbow" : [],
                "relbow" : [],
                "lhip" : [],
                "rhip" : [],
                "lknee" : [],
                "rknee" : [],
                "lankle" : [],
                "rankle" : []
            }
        }

        # Each element in this dictionary is a list of length n-1
        # These are all "first derviative" metrics like velocity
        self.velocity_metrics = {
            "student" : {
                "lshoulder" : [],
                "rShoulder" : [],
                "lelbow" : [],
                "relbow" : [],
                "lhip" : [],
                "rhip" : [],
                "lknee" : [],
                "rknee" : [],
                "lankle" : [],
                "rankle" : []
            },
            "teacher" : {
                "lshoulder" : [],
                "rShoulder" : [],
                "lelbow" : [],
                "relbow" : [],
                "lhip" : [],
                "rhip" : [],
                "lknee" : [],
                "rknee" : [],
                "lankle" : [],
                "rankle" : []
            }
        }

    def _calc_angle(self, joint, start_joint, end_joint):

        if joint[2]< 0.1 or start_joint[2]<0.1 or end_joint[2]<0.1:
            return -1

        # Calculate two vectors that form joint
        v1 = start_joint[0:2] - joint[0:2]
        v2 = end_joint[0:2] - joint[0:2]

        # Calc dot product
        dot_prod = np.dot(v1,v2)

        # Calculate magnitudes
        v1_mag = np.linalg.norm(v1)
        v2_mag = np.linalg.norm(v2)

        # Calculate angle
        if dot_prod/v1_mag/v2_mag > 1.:
            return np.arccos(1.)
        elif dot_prod/v1_mag/v2_mag < -1.:
            return np.arccos(-1.)
        return np.arccos(dot_prod/v1_mag/v2_mag)

    def _calc_velocity(self, prev_joint, cur_joint):

        if prev_joint[2]<0.1 or cur_joint[2]<0.1:
            return -1

        v1 = prev_joint[0:2]
        v2 = cur_joint[0:2]

        return np.linalg.norm(v2-v1)



    # Joint we are considering
    # For each of these, we calculate the angle and velocity
    # Left Shoulder
    # Left Elbow
    # Left Hip
    # Left Knee
    # Left Ankle
    # Right Shoulder
    # Right Elbow
    # Right Hip
    # Right Knee
    # Right Ankle

    def _calc_dance_metrics(self, dancer):
        # select data
        if(dancer != "student" and dancer != "teacher"):
            raise Exception("Selected dancer must be a student or teacher")

        # Create numpy arrays of the right length
        for joint in self.position_metrics[dancer]:
            self.position_metrics[dancer][joint] = np.zeros(shape = (len(self.poses[dancer]), ), dtype = np.float32)
            self.velocity_metrics[dancer][joint] = np.zeros(shape = (len(self.poses[dancer])-1, ), dtype = np.float32)

        for i, pose in enumerate(self.poses[dancer]):
            joint_angle_args = {
                "lshoulder" : [pose[0,Joint.LSHOULDER,:], pose[0,Joint.NECK,:], pose[0,Joint.LELBOW,:]],
                "rShoulder" : [pose[0,Joint.RSHOULDER,:], pose[0,Joint.NECK,:], pose[0,Joint.RELBOW,:]],
                "lelbow" : [pose[0,Joint.LELBOW,:], pose[0,Joint.LSHOULDER,:], pose[0,Joint.LWRIST,:]],
                "relbow" : [pose[0,Joint.RELBOW,:], pose[0,Joint.RSHOULDER,:], pose[0,Joint.RWRIST,:]],
                "lhip" : [pose[0,Joint.LHIP,:], pose[0,Joint.NECK,:], pose[0,Joint.LKNEE,:]],
                "rhip" : [pose[0,Joint.RHIP,:], pose[0,Joint.NECK,:], pose[0,Joint.RKNEE,:]],
                "lknee" : [pose[0,Joint.LKNEE,:], pose[0,Joint.LHIP,:], pose[0,Joint.LANKLE,:]],
                "rknee" : [pose[0,Joint.RKNEE,:], pose[0,Joint.RHIP,:], pose[0,Joint.RANKLE,:]],
                "lankle" : [pose[0,Joint.LANKLE,:], pose[0,Joint.LKNEE,:], pose[0,Joint.LBIGTOE,:]],
                "rankle" : [pose[0,Joint.RANKLE,:], pose[0,Joint.RKNEE,:], pose[0,Joint.RBIGTOE,:]]
            }

            # Calculate all of the joint angles and write them to the position metrics dictionary
            for joint, args in joint_angle_args.items():
                self.position_metrics[dancer][joint][i] = self._calc_angle(*args)


            if(i > 0):
                posePrev = self.poses[dancer][i-1]
                joint_vel_args = {
                    "lshoulder" : [posePrev[0,Joint.LSHOULDER,:], pose[0,Joint.LSHOULDER,:]],
                    "rShoulder" : [posePrev[0,Joint.RSHOULDER,:], pose[0,Joint.RSHOULDER,:]],
                    "lelbow" : [posePrev[0,Joint.LELBOW,:], pose[0,Joint.LELBOW,:]],
                    "relbow" : [posePrev[0,Joint.RELBOW,:], pose[0,Joint.RELBOW,:]],
                    "lhip" : [posePrev[0,Joint.LHIP,:], pose[0,Joint.LHIP,:]],
                    "rhip" : [posePrev[0,Joint.RHIP,:], pose[0,Joint.RHIP,:]],
                    "lknee" : [posePrev[0,Joint.LKNEE,:], pose[0,Joint.LKNEE,:]],
                    "rknee" : [posePrev[0,Joint.RKNEE,:], pose[0,Joint.RKNEE,:]],
                    "lankle" : [posePrev[0,Joint.LANKLE,:], pose[0,Joint.LANKLE,:]],
                    "rankle" : [posePrev[0,Joint.RANKLE,:], pose[0,Joint.RANKLE,:]]
                }

                for joint, args in joint_vel_args.items():
                    self.velocity_metrics[dancer][joint][i-1] = self._calc_velocity(*args)

    def add_frame_pose(self, student_pose, teacher_pose):
        """Add pose from a pair of frames from the student and teacher.

        Args:
            student_pose: A dict-type object that contains the the (x,y) coords of all of keypoints of the student
            teacher_pose: A dict-type object that contains the the (x,y) coords of all of keypoints of the teacher
        """

        self.poses["student"].append(student_pose)
        self.poses["teacher"].append(teacher_pose)


    def score_dancer(self):
        """Generates a score rating the quality of the dancer.

        Returns:
            A dictionary containing scores for individual limbs as well as an overall score
        """

        self._calc_dance_metrics("student")
        self._calc_dance_metrics("teacher")



        position_errors = {
                "lshoulder" : None,
                "rShoulder" : None,
                "lelbow" : None,
                "relbow" : None,
                "lhip" : None,
                "rhip" : None,
                "lknee" : None,
                "rknee" : None,
                "lankle" : None,
                "rankle" : None
            }

        velocity_errors = {
            "lshoulder" : None,
            "rShoulder" : None,
            "lelbow" : None,
            "relbow" : None,
            "lhip" : None,
            "rhip" : None,
            "lknee" : None,
            "rknee" : None,
            "lankle" : None,
            "rankle" : None
        }

        avg_position_errors = {
            "lshoulder" : None,
            "rShoulder" : None,
            "lelbow" : None,
            "relbow" : None,
            "lhip" : None,
            "rhip" : None,
            "lknee" : None,
            "rknee" : None,
            "lankle" : None,
            "rankle" : None
        }

        avg_velocity_errors = {
            "lshoulder" : None,
            "rShoulder" : None,
            "lelbow" : None,
            "relbow" : None,
            "lhip" : None,
            "rhip" : None,
            "lknee" : None,
            "rknee" : None,
            "lankle" : None,
            "rankle" : None
        }

        scores = {
            "lshoulder" : None,
            "rShoulder" : None,
            "lelbow" : None,
            "relbow" : None,
            "lhip" : None,
            "rhip" : None,
            "lknee" : None,
            "rknee" : None,
            "lankle" : None,
            "rankle" : None
        }

        for joint in position_errors:
            for i in range(self.position_metrics['student'][joint].shape[0]):
                if self.position_metrics['student'][joint][i]==-1 or self.position_metrics['teacher'][joint][i]==-1:
                    self.position_metrics['student'][joint][i] = 0
                    self.position_metrics['teacher'][joint][i] = 0
            position_errors[joint] = np.linalg.norm(np.expand_dims(self.position_metrics["student"][joint] - self.position_metrics["teacher"][joint], axis = 1), axis = 1)
            velocity_errors[joint] = np.linalg.norm(np.expand_dims(self.velocity_metrics["student"][joint] - self.velocity_metrics["teacher"][joint], axis = 1), axis = 1)

            avg_position_errors[joint] = np.average(position_errors[joint])
            avg_velocity_errors[joint] = np.average(velocity_errors[joint])

            sigma = DanceScorer.RANGE[joint]/DanceScorer.SIGMA_SCALE

            z = avg_position_errors[joint]/sigma
            scores[joint] = (-1*(norm.cdf(abs(z))*2-1))+1

        total = 0
        avg = 0

        for joint, score in scores.items():
            # Scale score by 2.5 to make it less disheartening
            # With the current scheme, the scores are very low, scale them up so they saturate the 0-100 spectrum better
            if(score != 1):
                avg += 2.5*score
                total += 1

        scores["average"] = avg/total

        return scores


if __name__ == "__main__":
    datasets = ["numpyfiles/caro-ymca.npy",
                "numpyfiles/caro1.npy",
                "numpyfiles/caro2.npy",
                "numpyfiles/david-null.npy",
                "numpyfiles/david-ymca.npy",
                "numpyfiles/FF-caro1.npy",
                "numpyfiles/FF-caro2.npy",
                "numpyfiles/null-ymca.npy"]
    # keypoints = np.squeeze(np.load("posekeypoints.npy"))

    joint_extremes = {
            "lshoulder" : {
                "min" : 5.0,
                "max" : 0.0
            },
            "rShoulder" : {
                "min" : 5.0,
                "max" : 0.0
            },
            "lelbow" : {
                "min" : 5.0,
                "max" : 0.0
            },
            "relbow" : {
                "min" : 5.0,
                "max" : 0.0
            },
            "lhip" : {
                "min" : 5.0,
                "max" : 0.0
            },
            "rhip" : {
                "min" : 5.0,
                "max" : 0.0
            },
            "lknee" : {
                "min" : 5.0,
                "max" : 0.0
            },
            "rknee" : {
                "min" : 5.0,
                "max" : 0.0
            },
            "lankle" : {
                "min" : 5.0,
                "max" : 0.0
            },
            "rankle" : {
                "min" : 5.0,
                "max" : 0.0
            }
        }

    for dataset in datasets:
        data = np.load(dataset)
        test = DanceScorer()
        print(data.shape)
        test.poses["student"] = data

        test._calc_dance_metrics("student")

        dataset_average = 0.0
        dataset_count = 0.0
        for joint in test.velocity_metrics["student"]:

            dataset_average += np.sum(test.velocity_metrics["student"][joint][test.velocity_metrics["student"][joint] >= 0])
            dataset_count += len(test.velocity_metrics["student"][joint][test.velocity_metrics["student"][joint] >= 0])

            temp_max_val = 0.0
            temp_min_val = 5.0

            temp_max_val = np.amax(test.velocity_metrics["student"][joint])

            if(len(test.velocity_metrics["student"][joint][test.velocity_metrics["student"][joint] >= 0]) > 0):
                temp_min_val = np.amin(test.velocity_metrics["student"][joint][test.velocity_metrics["student"][joint] >= 0])

            if(temp_max_val > joint_extremes[joint]["max"]):
                joint_extremes[joint]["max"] = temp_max_val

            if(temp_min_val < joint_extremes[joint]["min"]):
                joint_extremes[joint]["min"] = temp_min_val


        print("{} average velocity: {}".format(dataset, dataset_average/dataset_count))




    for joint in test.velocity_metrics["student"]:
        print("{}: {} {}".format(joint, joint_extremes[joint]["min"], joint_extremes[joint]["max"]))


    # test.poses['student'] = dancer
    # test.poses['teacher'] = teacher
    # test.score_dancer()
    # # test.add_frame_pose(keypoints, keypoints)
    # # test.add_frame_pose(keypoints, keypoints)

    # print(test.score_dancer())