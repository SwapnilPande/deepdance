

class DanceScorer:

    def __init__(self):

        # Instantiate two lists to store the teacher and student poses
        self.poses = {
            "student" : [],
            "teacher" : []
        }


        # Each element in this dictionary is a list of length n storing the tracked metrics
        self.position_metrics = {
            "student" : [],
            "teacher" : []
        }

        # Each element in this dictionary is a list of length n-1
        # These are all "first derviative" metrics like velocity
        self.velocity_metrics = {
            "student" : [],
            "teacher" : []
        }



    def add_frame_pose(self, student_pose, teacher_pose):
        """Add pose from a pair of frames from the student and teacher.

        Args:
            student_pose: A dict-type object that contains the the (x,y) coords of all of keypoints of the student
            teacher_pose: A dict-type object that contains the the (x,y) coords of all of keypoints of the teacher
        """

        pass


    def score_dancer():
        """Generates a score rating the quality of the dancer.

        Returns:
            A dictionary containing scores for individual limbs as well as an overall score
        """

        return poses








