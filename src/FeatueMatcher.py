import cv2
import numpy as np
from utils import draw_matches


class FeatureMatcher:
    def __init__(self):
        super().__init__()
        self.orb = cv2.ORB_create(nfeatures=200)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.last_keypoints = None
        self.last_descriptors = None
        self.last_translation = None  # Track last translation for relative scale

    def predict(self, prev_image_color, current_image_color, K):
        img_left = cv2.cvtColor(prev_image_color, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(current_image_color, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors with ORB
        keypoints_left, descriptors_left = self.orb.detectAndCompute(img_left, None)
        keypoints_right, descriptors_right = self.orb.detectAndCompute(img_right, None)

        # Match descriptors using knnMatch
        matches = self.bf.knnMatch(descriptors_left, descriptors_right, k=2)

        # Apply Lowe's ratio test
        ratio_threshold = 0.8
        good_matches = []

        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                p1 = keypoints_left[m.queryIdx].pt
                p2 = keypoints_right[m.trainIdx].pt

                if np.linalg.norm(np.array(p1) - np.array(p2)) < 0.1:
                    good_matches.append(m)

        if len(good_matches) < 8:
            print('Not enough matches')
            return None, None, 0

        # Prepare points for RANSAC
        src_pts = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        E, inliers = cv2.findEssentialMat(
            src_pts,
            dst_pts,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=0.1,
            maxIters=700
        )

        if inliers is None:
            print("No inliers found.")
            return None, None, 0

        _, R, t, inlier_mask = cv2.recoverPose(E, src_pts, dst_pts, cameraMatrix=K, mask=inliers)

        # Calculate relative scale based on previous translation
        estimated_scale = self.get_relative_scale(t)

        # Normalize and scale translation vector
        if np.linalg.norm(t) > 0:
            t /= np.linalg.norm(t)  # Normalize to unit vector
            t *= estimated_scale  # Scale with relative scale factor

        # Get transformation matrix
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = np.squeeze(t)

        # Draw matches on the combined image
        matched_image = draw_matches(
            prev_image_color, keypoints_left,
            current_image_color, keypoints_right,
            good_matches,
        )

        repeatability_score = self.calculate_repeatability_score(
            good_matches,
            keypoints_left,
            keypoints_right
        )

        return matched_image, T, repeatability_score

    def get_relative_scale(self, t):
        """Estimate relative scale based on the magnitude of translation vectors across frames."""
        if self.last_translation is None:
            # First frame, so no relative scale; assume initial scale of 1
            estimated_scale = 1.0
        else:
            # Estimate relative scale based on previous and current translation magnitudes
            last_magnitude = np.linalg.norm(self.last_translation)
            current_magnitude = np.linalg.norm(t)
            if last_magnitude > 0:
                estimated_scale = current_magnitude / last_magnitude
            else:
                estimated_scale = 1.0  # Default scale if last magnitude is zero

        # Update last translation vector
        self.last_translation = t
        return estimated_scale

    @staticmethod
    def calculate_repeatability_score(matches, kp1, kp2):
        num_matches = len(matches)
        num_kp1, num_kp2 = len(kp1), len(kp2)

        if num_kp1 > 0 and num_kp2 > 0:
            repeatability_score = num_matches / float(min(num_kp1, num_kp2))
            return repeatability_score * 100

        return 0.0
