import numpy as np

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
                "lankle" : []
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
                "lankle" : []
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
                "lankle" : []
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
                "lankle" : []
            }
        }

    def _calc_angle(self, joint, start_joint, end_joint):
        # Calculate two vectors that form joint
        v1 = start_joint[0:2] - joint[0:2]
        v2 = end_joint[0:2] - joint[0:2]

        # Calc dot product
        dot_prod = np.dot(v1,v2)

        # Calculate magnitudes
        v1_mag = np.linalg.norm(v1)
        v2_mag = np.linalg.norm(v2)

        # Calculate angle

        return np.arccos(dot_prod/v1_mag/v2_mag)

    def _calc_velocity(self, prev_joint, cur_joint):
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
                "lshoulder" : [pose[Joint.LSHOULDER,:], pose[Joint.NECK,:], pose[Joint.LELBOW,:]],
                "rShoulder" : [pose[Joint.RSHOULDER,:], pose[Joint.NECK,:], pose[Joint.RELBOW,:]],
                "lelbow" : [pose[Joint.LELBOW,:], pose[Joint.LSHOULDER,:], pose[Joint.LWRIST,:]],
                "relbow" : [pose[Joint.RELBOW,:], pose[Joint.RSHOULDER,:], pose[Joint.RWRIST,:]],
                "lhip" : [pose[Joint.LHIP,:], pose[Joint.NECK,:], pose[Joint.LKNEE,:]],
                "rhip" : [pose[Joint.RHIP,:], pose[Joint.NECK,:], pose[Joint.RKNEE,:]],
                "lknee" : [pose[Joint.LKNEE,:], pose[Joint.LHIP,:], pose[Joint.LANKLE,:]],
                "rknee" : [pose[Joint.RKNEE,:], pose[Joint.RHIP,:], pose[Joint.RANKLE,:]],
                "lankle" : [pose[Joint.LANKLE,:], pose[Joint.LKNEE,:], pose[Joint.RBIGTOE,:]]
            }

            # Calculate all of the joint angles and write them to the position metrics dictionary
            for joint, args in joint_angle_args.items():
                self.position_metrics[dancer][joint][i] = self._calc_angle(*args)


            if(i > 0):
                posePrev = self.poses[dancer][i-1]
                joint_vel_args = {
                    "lshoulder" : [posePrev[Joint.LSHOULDER,:], pose[Joint.LSHOULDER,:]],
                    "rShoulder" : [posePrev[Joint.RSHOULDER,:], pose[Joint.RSHOULDER,:]],
                    "lelbow" : [posePrev[Joint.LELBOW,:], pose[Joint.LELBOW,:]],
                    "relbow" : [posePrev[Joint.RELBOW,:], pose[Joint.RELBOW,:]],
                    "lhip" : [posePrev[Joint.LHIP,:], pose[Joint.LHIP,:]],
                    "rhip" : [posePrev[Joint.RHIP,:], pose[Joint.RHIP,:]],
                    "lknee" : [posePrev[Joint.LKNEE,:], pose[Joint.LKNEE,:]],
                    "rknee" : [posePrev[Joint.RKNEE,:], pose[Joint.RKNEE,:]],
                    "lankle" : [posePrev[Joint.RANKLE,:], pose[Joint.LANKLE,:]]
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
                "lankle" : None
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
            "lankle" : None
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
                "lankle" : None
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
            "lankle" : None
        }

        for joint in position_errors:
            position_errors[joint] = np.linalg.norm(np.expand_dims(self.position_metrics["student"][joint] - self.position_metrics["teacher"][joint], axis = 1), axis = 1)
            velocity_errors[joint] = np.linalg.norm(np.expand_dims(self.velocity_metrics["student"][joint] - self.velocity_metrics["teacher"][joint], axis = 1), axis = 1)

            avg_position_errors[joint] = np.average(position_errors[joint])
            avg_velocity_errors[joint] = np.average(velocity_errors[joint])

        return avg_position_errors, avg_velocity_errors


if __name__ == "__main__":
    keypoints = np.squeeze(np.load("posekeypoints.npy"))

    test = DanceScorer()
    test.add_frame_pose(keypoints, keypoints)
    test.add_frame_pose(keypoints, keypoints)

    print(test.score_dancer())






