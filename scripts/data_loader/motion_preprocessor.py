import numpy as np

class MotionPreprocessor:
    """
    Class to preprocess and filter motion data (skeletons) for both the AQGT and TED Expressive datasets.

    Attributes:
        dataset_name (str): Name of the dataset ('aqgt' or 'TED_Expressive').
        skeletons (np.array): The array of skeleton poses for the sample.
        mean_pose (np.array): The mean pose used for normalization.
        words (list): List of words associated with the motion data.
        filtering_message (str): Message indicating the reason for filtering the sample.
    """
    
    def __init__(self, dataset_name, skeletons, mean_pose, words):
        """
        Initializes the MotionPreprocessor with the dataset type, skeleton data, and mean pose.

        Args:
            dataset_name (str): Name of the dataset ('aqgt' or 'TED_Expressive').
            skeletons (np.array): Array of skeleton poses for the sample.
            mean_pose (np.array): Mean pose for skeleton normalization.
            words (list): List of words associated with the motion data.
        """
        self.dataset_name = dataset_name
        self.skeletons = np.array(skeletons)
        self.mean_pose = np.array(mean_pose).reshape(-1, 3)
        self.words = words
        self.filtering_message = "PASS"  # Default message indicating no filtering

    def get(self):
        """
        Applies various filtering checks on the skeleton data, such as pose differences, 
        spine angle, and static motion.

        Returns:
            tuple: Filtered skeletons and filtering message.
        """
        assert self.skeletons is not None

        # Apply filtering checks on the skeletons
        if self.skeletons != []:
            if self.check_pose_diff():
                self.skeletons = []
                self.filtering_message = "pose"
            elif self.check_spine_angle():
                self.skeletons = []
                self.filtering_message = "spine angle"
            elif self.check_static_motion():
                self.skeletons = []
                self.filtering_message = "motion"
            elif self.check_words():
                self.skeletons = []
                self.filtering_message = "text"

        # Check for missing joints and convert skeletons to list format
        if self.skeletons != []:
            self.skeletons = self.skeletons.tolist()
            for i in self.skeletons:
                assert not np.isnan(i).any()  # Ensure no missing joints

        return self.skeletons, self.filtering_message
    
    def check_words(self):
        """
        Filters the skeleton data based on the number of associated words.

        Returns:
            bool: True if the word count exceeds the thresholds, otherwise False.
        """
        if len(self.words) > 15:
            return True
        for sample in self.words:
            if len(sample[0]) > 20:
                return True
        return False

    def check_static_motion(self, verbose=False):
        """
        Checks if the skeleton data represents static motion by calculating the variance of joint movement.

        Args:
            verbose (bool): If True, prints the variance values for debugging.

        Returns:
            bool: True if the motion is static (below a variance threshold), otherwise False.
        """
        if self.dataset_name == "aqgt":
            def get_variance(skeleton):
                variance = np.median(np.var(skeleton.flatten()))
                return variance

            body_var = get_variance(self.skeletons)
            th = 0.006  # Threshold for detecting static motion in AQGT
            if body_var < th:
                if verbose:
                    print('skip - check_static_motion body var {}'.format(body_var))
                return True
            else:
                if verbose:
                    print('pass - check_static_motion body var {}'.format(body_var))
                return False
            
        elif self.dataset_name == "TED_Expressive":
            def get_variance(skeleton, joint_idx):
                wrist_pos = skeleton[:, joint_idx]
                variance = np.sum(np.var(wrist_pos, axis=0))
                return variance

            left_arm_var = get_variance(self.skeletons, 6)  # Left wrist joint
            right_arm_var = get_variance(self.skeletons, 7)  # Right wrist joint

            th = 0.0014  # Threshold for detecting static motion in TED Expressive
            if left_arm_var < th and right_arm_var < th:
                if verbose:
                    print('skip - check_static_motion left var {}, right var {}'.format(left_arm_var, right_arm_var))
                return True
            else:
                if verbose:
                    print('pass - check_static_motion left var {}, right var {}'.format(left_arm_var, right_arm_var))
                return False

    def check_pose_diff(self, verbose=False):
        """
        Checks if the difference between the current skeleton and the mean pose is below a threshold.

        Args:
            verbose (bool): If True, prints the difference value for debugging.

        Returns:
            bool: True if the difference is below the threshold, otherwise False.
        """
        diff = np.abs(self.skeletons - self.mean_pose)
        diff = np.mean(diff)

        th = 0.02  # Threshold for pose difference
        if diff < th:
            if verbose:
                print('skip - check_pose_diff {:.5f}'.format(diff))
            return True
        else:
            if verbose:
                print('pass - check_pose_diff {:.5f}'.format(diff))
            return False

    def check_spine_angle(self, verbose=False):
        """
        Checks if the spine angle is within an acceptable range based on the dataset.

        Args:
            verbose (bool): If True, prints the spine angle values for debugging.

        Returns:
            bool: True if the spine angle is outside the acceptable range, otherwise False.
        """
        def angle_between(v1, v2):
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        angles = []
        for i in range(self.skeletons.shape[0]):
            if self.dataset_name == "aqgt":
                spine_vec = self.skeletons[i, 2] - self.skeletons[i, 0]  # AQGT dataset spine calculation
            elif self.dataset_name == "TED_Expressive":
                spine_vec = self.skeletons[i, 1] - self.skeletons[i, 0]  # TED Expressive dataset spine calculation
            angle = angle_between(spine_vec, [0, -1, 0])
            angles.append(angle)

        # Check if the spine angle exceeds acceptable thresholds
        if np.rad2deg(max(angles)) > 30 or np.rad2deg(np.mean(angles)) > 20:
            if verbose:
                print('skip - check_spine_angle {:.5f}, {:.5f}'.format(max(angles), np.mean(angles)))
            return True
        else:
            if verbose:
                print('pass - check_spine_angle {:.5f}'.format(max(angles)))
            return False
