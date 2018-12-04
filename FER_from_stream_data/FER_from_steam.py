import cv2

face_detector = cv2.CascadeClassifier('../venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

capture_stream = cv2.VideoCapture(0)
