import cv2
from keras.models import model_from_json

model_in_json = open('./models/facial_expression_model_structure.json', 'r').read()
FER_model = model_from_json(model_in_json)
FER_model.load_weights('./models/facial_expression_model_weights.h5')

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
face_detector = cv2.CascadeClassifier('../venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

capture_stream = cv2.VideoCapture(0)
capture_stream.set(3, 640)  # width
capture_stream.set(4, 480)  # height

while True:
    ret, frame = capture_stream.read()

    image_in_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        image_in_grayscale,
        scaleFactor=1.1,
        minNeighbors=10,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(3, 3))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Rectangle around the face
        cv2.imshow('Stream from the camera', frame)

        detected_face = frame[y:y + h, x:x + w]  # crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48x48

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture_stream.release()
cv2.destroyAllWindows()
