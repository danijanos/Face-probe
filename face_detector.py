import cv2
import glob

# The four open cv HAAR filter face detector:
face_detector_1 = cv2.CascadeClassifier("venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
face_detector_2 = cv2.CascadeClassifier("venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml")
face_detector_3 = cv2.CascadeClassifier("venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml")
face_detector_4 = cv2.CascadeClassifier("venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt_tree.xml")

# Define emotions:
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]


def detect_faces(face_emotion):
    # Get a list of all images with the specified emotion:
    files = glob.glob("sorted_set\\%s\\*" % face_emotion)
    file_number = 0
    for file in files:
        # Open image:
        face_image = cv2.imread(file)
        # Convert image to gray scale:
        grayscale_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        # Detect the face using the 4 different classifiers:
        face_1 = face_detector_1.detectMultiScale(
            grayscale_face_image,
            scaleFactor=1.1,
            minNeighbors=10,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(5, 5)
        )

        face_2 = face_detector_2.detectMultiScale(
            grayscale_face_image,
            scaleFactor=1.1,
            minNeighbors=10,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(5, 5)
        )

        face_3 = face_detector_3.detectMultiScale(
            grayscale_face_image,
            scaleFactor=1.1,
            minNeighbors=10,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(5, 5)
        )

        face_4 = face_detector_4.detectMultiScale(
            grayscale_face_image,
            scaleFactor=1.1,
            minNeighbors=10,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(5, 5)
        )

        # Go over detected faces, stop at first detected face, return empty if no face:
        if len(face_1) == 1:
            face_features = face_1
        elif len(face_2) == 1:
            face_features = face_2
        elif len(face_3) == 1:
            face_features = face_3
        elif len(face_4) == 1:
            face_features = face_4
        else:
            face_features = ""

        # Cut and save the face:
        # get the coordinates and size of the rectangle which containing the face
        for (x, y, w, h) in face_features:
            print("face found in file: %s" % file)
            # Cut the image frame to size:
            grayscale_face_image = grayscale_face_image[y:y + h, x:x + w]
            try:
                # Resize all images to have the same size:
                cropped_and_resized_image = cv2.resize(grayscale_face_image, (350, 350))
                # Write out:
                cv2.imwrite("dataset\\%s\\%s.jpg" % (face_emotion, file_number), cropped_and_resized_image)
            except:
                # If error, pass the file
                pass
        # Increment image number:
        file_number += 1


for emotion in emotions:
    detect_faces(emotion)
