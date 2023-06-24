# -*- coding: cp949 -*- 

import cv2
import numpy as np
import boto3
from keras.models import model_from_json
from keras.preprocessing import image
from keras.utils.image_utils import img_to_array
import os

aws_access_key_id = os.environ.get("aws_access_key_id")
aws_secret_access_key = os.environ.get("aws_secret_access_key")
aws_bucket_name = os.environ.get("aws_bucket_name")

# Loading JSON model
json_file = open('top_models/fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Loading weights
model.load_weights('top_models/fer.h5')

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def download_file_from_s3(bucket_name, s3_key, local_path, aws_access_key_id, aws_secret_access_key):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    try:
        s3.download_file(bucket_name, s3_key, local_path)
        print("파일 다운로드 완료:", local_path)
    except Exception as e:
        print("파일 다운로드 실패:", str(e))

def perform_emotion_recognition(video_path, start_times):
    print(start_times)

    if isinstance(start_times, float):
        start_times = [start_times]  # Convert float to list

    cap = cv2.VideoCapture(video_path)

    emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
    emotion_values = {}

    for start_time in start_times:
        end_time = start_time + 20

        total_emotion_values = np.zeros(8)  # Initialize an array to store total emotion values
        frame_count = 0
        face_detected = False

        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)  # Set the starting point of the video

        while True:
            ret, img = cap.read()
            if not ret or cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 > end_time:
                break

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.2, 6)

            if len(faces_detected) > 0:
                face_detected = True

            for (x, y, w, h) in faces_detected:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
                roi_gray = gray_img[y:y + w, x:x + h]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255.0

                predictions = model.predict(img_pixels)
                max_index = np.argmax(predictions[0])

                total_emotion_values[max_index] += 1
                predicted_emotion = emotions[max_index]

            frame_count += 1

        if not face_detected:
            total_frames = 1
        else:
            total_frames = frame_count

        emotion_ratios = total_emotion_values / total_frames

        emotion_values[start_time] = {}
        for emotion, ratio in zip(emotions, emotion_ratios):
            emotion_values[start_time][emotion] = ratio * 100

    return emotion_values