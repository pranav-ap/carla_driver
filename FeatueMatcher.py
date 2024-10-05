import threading
import cv2
from utils import draw_matches


class FeatureMatcher:
    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.orb = cv2.ORB_create(100)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def predict(self, prev_image_color, current_image_color):
        prev_image_gray = cv2.cvtColor(prev_image_color, cv2.COLOR_RGB2GRAY)
        current_image_gray = cv2.cvtColor(current_image_color, cv2.COLOR_RGB2GRAY)

        with self.lock:
            kp1, des1 = self.orb.detectAndCompute(prev_image_gray, None)
            kp2, des2 = self.orb.detectAndCompute(current_image_gray, None)

            matches = self.bf.match(des1, des2)

        matches = sorted(matches, key=lambda x: x.distance)

        num_matches_to_draw = min(30, len(matches))
        some_matches = matches[:num_matches_to_draw]

        matched_image = draw_matches(
            prev_image_color, kp1,
            current_image_color, kp2,
            some_matches
        )

        return matched_image
