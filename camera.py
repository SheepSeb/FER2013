import cv2
import numpy as np
import tensorflow as tf
from keras import models
from keras.preprocessing import image

# model = models.load_model('model3.hdf5')
model = models.load_model("emotionModel.hdf5")
# model = models.load_model("model_2.hdf5")

face_classifier = cv2.CascadeClassifier('har.xml')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64))

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = tf.keras.preprocessing.image.img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi)[0]
            print(preds)

            label = np.argmax(preds)
            label_map = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            label_text = label_map[label]
            cv2.putText(frame, label_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
exit()