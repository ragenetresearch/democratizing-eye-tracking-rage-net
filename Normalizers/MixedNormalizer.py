import os

import cv2
import dlib
import numpy as np
import scipy.io as sio
from imutils import face_utils

from Normalizers import NormalizationUtils


class MixedNormalizer:
    def __init__(self, eye_roi_size=(60, 36), eyes_together=False, face_grid_size=25, equalization='Hist', base_path='.'):
        """
        Initialize normalizer instance.
        ---
        :eye_roi_size - size of eye-rois
        :eyes_together - normalize both eyes separately or together as 1 img?
        :face_grid_size - grid resolutions, where will be 4 face_grid coordinates Calculated (e.g. 25 means 25x25 grid)
        :equalization - 'Hist' | 'Clahe' | 'None
        """
        model_file = os.path.join(base_path, 'Models', 'OpencvDNN', 'res10_300x300_ssd_iter_140000.caffemodel')
        model_config_file = os.path.join(base_path, 'Models', 'OpencvDNN', 'deploy.prototxt.txt')

        # Load face detector
        self.detector = cv2.dnn.readNetFromCaffe(model_config_file, model_file)

        # Predictor for detecting eye landmarks
        self.predictor = dlib.shape_predictor(
            os.path.join(base_path, 'ShapePredictors', 'shape_predictor_68_face_landmarks.dat'))

        # Detector's confidence interval
        self.confidence_threshold = 0.5

        # Camera calibration parameters
        self.camera_calibration = None
        self.camera_matrix = None
        self.camera_distortion = None

        # Normalization parameters #

        # Focal length of normalized camera
        self.focal_length_norm = 960

        # Normalized distance between eye and camera
        self.distance_norm = 600

        # Size of cropped eye
        self.eye_roi_size = eye_roi_size

        # If eyes should be together as 1 image or each eye individually
        self.eyes_together = eyes_together

        # Generic 3D face coordinates for normalization
        self.generic_3d_face_coordinates = np.array([
            [-45.0967681126441, -21.3128582097374, 21.3128582097374,
             45.0967681126441, -26.2995769055718, 26.2995769055718],
            [-0.483773045049757, 0.483773045049757, 0.483773045049757,
             -0.483773045049757, 68.5950352778326, 68.5950352778326],
            [2.39702984214363, -2.39702984214363, -2.39702984214363,
             2.39702984214363, -9.86076131526265 * (10 ** -32), -9.86076131526265 * (10 ** -32)],
        ])
        self.generic_3d_face_coordinates_T = self.generic_3d_face_coordinates.T

        # Grid size, e.g. [25 x 25]
        self.face_grid_size = face_grid_size

        # Equalization type ('Hist' || 'Clahe' || None)
        self.equalization = equalization

    def load_calibration_parameters(self, camera_file):
        self.camera_calibration = sio.loadmat(camera_file)
        self.camera_matrix = self.camera_calibration['cameraMatrix']
        self.camera_distortion = self.camera_calibration['distCoeffs']

    def set_calibration_parameters(self, camera_matrix, camera_distortion):
        self.camera_matrix = camera_matrix
        self.camera_distortion = camera_distortion

    def undistort_image(self, img):
        """
        Undistort image using cv2.undistort function.
        Resources:
         - https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        """
        return cv2.undistort(img, self.camera_matrix, self.camera_distortion)

    def estimate_head_pose(self, face_2d):
        """
        Retrieve rotation and translation vectors from 2D to generic 3D face model projections.
        Resources:
         - https://www.pythonpool.com/opencv-solvepnp/
         - https://www.morethantechnical.com/2012/10/17/head-pose-estimation-with-opencv-opengl-revisited-w-code/
         - https://stackoverflow.com/questions/36590516/how-to-get-3d-coordinate-axes-of-head-pose-estimation-in-dlib-c/36591123#36591123
         - https://stackoverflow.com/questions/53298065/opencv-confusion-about-solvepnp
        """
        _, rot_vec, trans_vec = cv2.solvePnP(self.generic_3d_face_coordinates_T,
                                             face_2d,
                                             self.camera_matrix,
                                             self.camera_distortion,
                                             flags=cv2.SOLVEPNP_EPNP)
        # Iterate PnP again
        _, rot_vec, trans_vec = cv2.solvePnP(self.generic_3d_face_coordinates_T,
                                             face_2d,
                                             self.camera_matrix,
                                             self.camera_distortion,
                                             rot_vec,
                                             trans_vec,
                                             True)

        return rot_vec, trans_vec

    def retrieve_eyes(self, rot_vec, trans_vec):
        head_translation = trans_vec.reshape((3, 1))
        head_rotation = cv2.Rodrigues(rot_vec)[0]

        # Get the 2D Coordinates
        face_landmarks_3d = np.dot(head_rotation, self.generic_3d_face_coordinates) + head_translation

        # Center of right eye (!Warn image is not mirrored)
        right_eye = 0.5 * (face_landmarks_3d[:, 0] + face_landmarks_3d[:, 1]).reshape((3, 1))

        # Center of left eye
        left_eye = 0.5 * (face_landmarks_3d[:, 2] + face_landmarks_3d[:, 3]).reshape((3, 1))

        return [right_eye, left_eye], head_rotation

    def retrieve_eyes_together(self, rot_vec, trans_vec):
        head_translation = trans_vec.reshape((3, 1))
        head_rotation = cv2.Rodrigues(rot_vec)[0]

        # Get the 2D Coordinates
        face_landmarks_3d = np.dot(head_rotation, self.generic_3d_face_coordinates) + head_translation

        # Center of eyes together
        eyes_center = 0.5 * (face_landmarks_3d[:, 0] + face_landmarks_3d[:, 3]).reshape((3, 1))

        return eyes_center, head_rotation

    def normalize_eye(self, eye, head_rotation, img_gray):
        """
        Normalize perspective of image, equalize histogram and return normalized eye roi.
        Resources:
         - https://theailearner.com/tag/cv2-warpperspective/
         - https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
        """
        distance = np.linalg.norm(eye)
        z_scale = self.distance_norm / distance

        # Camera matrix normalized
        camera_norm = np.array([
            [self.focal_length_norm, 0, self.eye_roi_size[0] / 2],
            [0, self.focal_length_norm, self.eye_roi_size[1] / 2],
            [0, 0, 1.0],
        ])

        # Scaling matrix S
        scaling_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ])

        # Calculate perspective transformation
        forward = (eye / distance).reshape(3)
        down = np.cross(forward, head_rotation[:, 0])  # Second arg. is head_rotation X
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)

        # Rotation matrix R
        rotation_matrix = np.c_[right, down, forward].T

        # Transformation matrix W
        transformation_matrix = np.dot(
            np.dot(camera_norm, scaling_matrix),
            np.dot(rotation_matrix, np.linalg.inv(self.camera_matrix)))

        # Normalize perspective
        img_warped = cv2.warpPerspective(img_gray, transformation_matrix, self.eye_roi_size)

        # Equalize histogram
        if self.equalization == 'Hist':
            img_warped = cv2.equalizeHist(img_warped)
        elif self.equalization == 'Clahe':
            img_warped = self.clahe.apply(img_warped)

        return img_warped

    def detect_face(self, img):
        """
        Detect face using OpenCV DNN caffe face detection model.
        Resources:
         - https://towardsdatascience.com/face-detection-models-which-to-use-and-why-d263e82c302c
         - https://stackoverflow.com/questions/37215036/dlib-vs-opencv-which-one-to-use-when
         - https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
         - https://towardsdatascience.com/face-detection-models-which-to-use-and-why-d263e82c302c
         - https://dev.to/azure/opencv-detect-and-blur-faces-using-dnn-40ab
        """
        img_height, img_width = img.shape[:2]
        img_blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))

        self.detector.setInput(img_blob)

        faces = self.detector.forward()
        for idx in range(faces.shape[2]):
            confidence = faces[0, 0, idx, 2]

            # Keep only face with confidence above threshold
            if confidence > self.confidence_threshold:
                x1 = faces[0, 0, idx, 3]
                y1 = faces[0, 0, idx, 4]
                x2 = faces[0, 0, idx, 5]
                y2 = faces[0, 0, idx, 6]

                # Create bounding box
                bounding_box = dlib.rectangle(int(x1 * img_width), int(y1 * img_height),
                                              int(x2 * img_width), int(y2 * img_height))

                # Create face grid landmarks (normalized - between <0;1>)
                face_grid_2d = [
                    [x1, y1],
                    [x2, y1],
                    [x1, y2],
                    [x2, y2],
                ]

                # Convert 2D landmarks to numpy array
                face_grid_2d = np.array(face_grid_2d, dtype=np.float64)

                return bounding_box, face_grid_2d

        raise Exception("[MixedNormalizer] - No face could be detected!")

    def normalize_image(self, file_name):
        # Load file
        img = cv2.imread(file_name)

        # Undistort image
        img = self.undistort_image(img)

        # Retrieve face and face grid 2D
        bounding_box, face_grid_2d = self.detect_face(img)

        # Image to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Process face_grid
        face_grid = NormalizationUtils.process_face_grid(face_grid_2d, self.face_grid_size)

        # Estimate 2D face landmarks
        # Note: Landmarks are indexed with 1 in image, index in python starts with 0
        landmarks = face_utils.shape_to_np(self.predictor(img, bounding_box))
        face_2d = np.array([landmarks[36],  # Left eye - left
                            landmarks[39],  # Left eye - right
                            landmarks[42],  # Right eye - left
                            landmarks[45],  # Right eye - right
                            landmarks[48],  # Mouse left
                            landmarks[54],  # Mouse right
                            ], dtype=np.float64)

        # Retrieve rotation and translation matrices
        rot_vec, trans_vec = self.estimate_head_pose(face_2d)

        # Normalize eyes together
        if self.eyes_together:
            eyes_center, head_rotation = self.retrieve_eyes_together(rot_vec, trans_vec)
            processed_eyes = self.normalize_eye(eyes_center, head_rotation, img)
            eye_screen_distance = np.linalg.norm(eyes_center)

            # EYE Rois, face_grid, distance from eyes
            return processed_eyes, face_grid, eye_screen_distance

        # Normalize and return each eye individually
        else:
            eyes, head_rotation = self.retrieve_eyes(rot_vec, trans_vec)
            processed_eyes = []
            eye_screen_distance = np.mean(np.array(np.linalg.norm(eyes[0]), np.linalg.norm(eyes[1])))

            for eye in eyes:
                processed_eyes.append(self.normalize_eye(eye, head_rotation, img))

            # EYE Rois, face_grid, distance from eyes
            return processed_eyes, face_grid, eye_screen_distance
