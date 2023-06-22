import argparse
import cv2
import numpy as np
from keras.models import model_from_json
from tensorflow import keras
img_to_array = keras.preprocessing.image.img_to_array

# Parse the video file path argument
ap = argparse.ArgumentParser()
ap.add_argument('video', help='Path to the input video file')
args = vars(ap.parse_args())

# Load JSON model
json_file = open('top_models/fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load weights
model.load_weights('top_models/fer.h5')

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(args['video'])

total_emotion_values = np.zeros(8)  # Initialize an array to store total emotion values
frame_count = 0  # Initialize frame count

while True:
    ret, img = cap.read()
    if not ret:
        break

    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current time in seconds

    # Define the specific time interval for emotion recognition (e.g., 5 seconds to 1 minute 30 seconds)
    start_time = 5
    end_time = 1 * 60 + 30

    if start_time <= current_time <= end_time:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.2, 6)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            roi_gray = gray_img[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255.0

            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])

            total_emotion_values[max_index] += 1  # Accumulate the emotion values

            emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
            predicted_emotion = emotions[max_index]

            cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    resized_img = cv2.resize(img, (1024, 768))
    cv2.imshow('Facial Emotion Recognition', resized_img)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Calculate emotion ratios
total_frames = frame_count
emotion_ratios = total_emotion_values / total_frames

# Print the emotion ratios
emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
for emotion, ratio in zip(emotions, emotion_ratios):
    print(f"{emotion}: {ratio:.2%}")
