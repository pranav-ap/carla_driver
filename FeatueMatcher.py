import threading
import cv2
import numpy as np

from utils import draw_matches


class FeatureMatcher:
    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()

        self.orb = cv2.ORB_create(150)

        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,
            key_size=12,
            multi_probe_level=1
        )
        search_params = dict(checks=50)

        self.flann = cv2.FlannBasedMatcher(
            indexParams=index_params,
            searchParams=search_params
        )

    @staticmethod
    def calculate_repeatability_score(matches, kp1, kp2):
        match_count = len(matches)
        # total_keypoints = min(len(kp1), len(kp2))
        total_keypoints = len(kp1)

        if total_keypoints == 0:
            return 0.0

        repeatability_score = match_count / total_keypoints

        return repeatability_score

    def predict_bf(self, prev_image_color, current_image_color):
        prev_image_gray = cv2.cvtColor(prev_image_color, cv2.COLOR_RGB2GRAY)
        current_image_gray = cv2.cvtColor(current_image_color, cv2.COLOR_RGB2GRAY)

        with self.lock:
            kp1, des1 = self.orb.detectAndCompute(prev_image_gray, None)
            kp2, des2 = self.orb.detectAndCompute(current_image_gray, None)

            matches = self.bf.match(des1, des2)

        matches = sorted(matches, key=lambda x: x.distance)

        q1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        q2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        num_matches_to_draw = min(30, len(matches))
        some_matches = matches[:num_matches_to_draw]

        matched_image = draw_matches(
            prev_image_color, kp1,
            current_image_color, kp2,
            some_matches
        )

        repeatability_score = self.calculate_repeatability_score(matches, kp1, kp2)

        return matched_image, kp1, des1, q1, kp2, des2, q2, repeatability_score

    def predict_flann(self, prev_image_color, current_image_color):
        prev_image_gray = cv2.cvtColor(prev_image_color, cv2.COLOR_RGB2GRAY)
        current_image_gray = cv2.cvtColor(current_image_color, cv2.COLOR_RGB2GRAY)

        with self.lock:
            kp1, des1 = self.orb.detectAndCompute(prev_image_gray, None)
            kp2, des2 = self.orb.detectAndCompute(current_image_gray, None)

            all_matches = self.flann.knnMatch(des1, des2, k=2)

            # Apply ratio test to filter matches
            matches = []

            for x in all_matches:
                if len(x) != 2:
                    continue

                m, n = x
                if m.distance < 0.8 * n.distance:
                    matches.append(m)

        matches = sorted(matches, key=lambda x: x.distance)

        q1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        q2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        num_matches_to_draw = min(15, len(matches))
        some_matches = matches[:num_matches_to_draw]

        matched_image = draw_matches(
            prev_image_color, kp1,
            current_image_color, kp2,
            some_matches
        )

        repeatability_score = self.calculate_repeatability_score(matches, kp1, kp2)

        return matched_image, kp1, des1, q1, kp2, des2, q2, repeatability_score
