import threading
import cv2
import numpy as np
import torch
from PIL import Image
from utils import draw_matches
import time


class FeatureMatcher:
    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()

        from transformers import AutoImageProcessor, SuperPointForKeypointDetection
        self.processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
        self.model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.orb = cv2.ORB_create()
        self.fast = cv2.FastFeatureDetector_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        FLANN_INDEX_LSH = 6
        self.flann = cv2.FlannBasedMatcher(
            indexParams=dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=6,
                key_size=12,
                multi_probe_level=1
            ),
            searchParams=dict(checks=50)
        )

    @staticmethod
    def calculate_repeatability_score(matches, kp1, kp2):
        match_count = len(matches)
        total_keypoints = min(len(kp1), len(kp2))

        if total_keypoints == 0:
            return 0.0

        repeatability_score = match_count / total_keypoints

        return repeatability_score

    def predict_classical(self, prev_image_color, current_image_color):
        prev_image_gray = cv2.cvtColor(prev_image_color, cv2.COLOR_RGB2GRAY)
        current_image_gray = cv2.cvtColor(current_image_color, cv2.COLOR_RGB2GRAY)

        with self.lock:
            kp1 = self.fast.detect(prev_image_gray, None)
            kp2 = self.fast.detect(current_image_gray, None)

            kp1 = sorted(kp1, key=lambda x: -x.response)
            kp2 = sorted(kp2, key=lambda x: -x.response)

            min_keypoints = min(len(kp1), 500)
            kp1 = kp1[:min_keypoints]

            min_keypoints = min(len(kp2), 500)
            kp2 = kp2[:min_keypoints]

            kp1, des1 = self.orb.compute(prev_image_gray, kp1)
            kp2, des2 = self.orb.compute(current_image_gray, kp2)

        all_matches = self.flann.knnMatch(des1, des2, k=2)

        # Apply ratio test to filter matches
        matches = []

        for x in all_matches:
            if len(x) != 2:
                continue

            m, n = x
            if m.distance < 0.8 * n.distance:
                matches.append(m)

        # Sort matches based on distance
        # matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched points from both images
        q1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        q2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        matched_image = draw_matches(
            prev_image_color, kp1,
            current_image_color, kp2,
            matches
        )

        repeatability_score = self.calculate_repeatability_score(matches, kp1, kp2)

        return matched_image, kp1, des1, q1, kp2, des2, q2, repeatability_score

    def extract_keypoints(self, outputs, index):
        image_mask = outputs.mask[index]
        indices = torch.nonzero(image_mask).squeeze()

        if indices.numel() == 0:
            return [], None, None

        keypoints = outputs.keypoints[index][indices]
        descriptors = outputs.descriptors[index][indices]
        scores = outputs.scores[index][indices]

        keypoints_cv = []
        for i, keypoint in enumerate(keypoints):
            kp = cv2.KeyPoint(x=keypoint[0].item(), y=keypoint[1].item(), size=1, response=scores[i].item())
            keypoints_cv.append(kp)

        descriptors = descriptors.detach().cpu().numpy().astype('uint8')

        return keypoints_cv, descriptors, scores

    def predict_sp(self, prev_image_color, current_image_color):
        images = [Image.fromarray(prev_image_color), Image.fromarray(current_image_color)]

        with self.lock:
            feature_start_time = time.time()
            print('start')
            inputs = self.processor(images, return_tensors="pt")

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            outputs = self.model(**inputs)
            elapsed_feature_time = time.time() - feature_start_time
            print(f'end {elapsed_feature_time}')

        # Extract keypoints and descriptors from both images
        keypoints1, descriptors1, scores1 = self.extract_keypoints(outputs, 0)
        keypoints2, descriptors2, scores2 = self.extract_keypoints(outputs, 1)

        # Use FLANN-based matcher for matching descriptors
        all_matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test to filter matches
        matches = []

        for x in all_matches:
            if len(x) != 2:
                continue

            m, n = x
            if m.distance < 0.7 * n.distance:
                matches.append(m)

        # Sort matches based on distance
        # matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched points from both images
        q1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        q2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        matched_image = draw_matches(
            prev_image_color, keypoints1,
            current_image_color, keypoints2,
            matches
        )

        repeatability_score = self.calculate_repeatability_score(matches, keypoints1, keypoints2)

        return matched_image, keypoints1, descriptors1, q1, keypoints2, descriptors2, q2, repeatability_score
