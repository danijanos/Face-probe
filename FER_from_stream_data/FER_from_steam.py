import cv2

face_detector = cv2.CascadeClassifier('../venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

capture_stream = cv2.VideoCapture(0)

while True:
    ret, img = capture_stream.read()

    img_in_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        img_in_grayscale,
        scaleFactor=1.1,
        minNeighbors=10,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(3, 3))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('img', img)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture_stream.release()
cv2.destroyAllWindows()
