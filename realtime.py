import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('maskdedector.keras')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Fixes the inverted camera

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (64, 64))
        arr = np.expand_dims(face_resized, axis=0).astype("float32")
        arr = arr / 255.0  # Make sure to normalize!
        pred = model.predict(arr)
        label = "No Mask" if pred[0][0] > 0.5 else "Mask"
        color = (0, 0, 255) if label == "No Mask" else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)

    cv2.imshow('Real-time Mask Detection (fixed)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
