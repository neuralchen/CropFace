import dlib
import cv2
import dlib
import numpy as np


if __name__ == "__main__":
    image = cv2.imread("C:/Users/chenx/Pictures/wocao2.jpg")
    detector  = dlib.get_frontal_face_detector()
    faces     = detector(image)
    print(faces)
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    landmarks  = predictor(image, faces[0])
    left_eye  = (landmarks.part(36).x,landmarks.part(36).y)
    right_eye  = (landmarks.part(45).x,landmarks.part(45).y)
    angle     = np.degrees(np.arctan2(right_eye[1]-left_eye[1], right_eye[0]-left_eye[0]))
    print(angle)