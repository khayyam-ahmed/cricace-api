import tensorflow as tf
import cv2
import numpy as np
from collections import Counter

# Define the class labels
class_labels = ['Cover Shot', 'Flick Shot', 'Pull Shot', 'Straight Shot', 'Sweep_Slog Shot']

# Load the model into the app.

"""UPDATE ACCORDINGLY WITH DOCKER FILE SYSTEM PATH"""
model = tf.keras.models.load_model("model/battingShotClassification/batting_style_classification.h5", compile=False)

def classifyBattingShot(video: cv2.VideoCapture):
    predicted_labels = []
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0

        frame = np.expand_dims(frame, axis=0)

        predicted_class = model.predict(frame)
        predicted_class_index = np.argmax(predicted_class, axis=-1)[0]

        predicted_label = class_labels[predicted_class_index]
        predicted_labels.append(predicted_label)

    print(predicted_labels)
    most_common_element = Counter(predicted_labels).most_common(1)[0][0]
    return {'Batting style': most_common_element}
