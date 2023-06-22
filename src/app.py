# -*- coding: cp949 -*- 

import json
import boto3
from botocore.exceptions import NoCredentialsError
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from keras.models import model_from_json
from keras.utils.image_utils import img_to_array
import show_result, amazon_transcribe
import os

# Flask ��ü �ν��Ͻ� ����
app = Flask(__name__)

# Amazon S3 ���� - ���� ���� ���
aws_access_key_id = os.environ.get("aws_access_key_id")
aws_secret_access_key = os.environ.get("aws_secret_access_key")
aws_region_name = os.environ.get("aws_region_name")
aws_bucket_name = os.environ.get("aws_bucket_name")


# ���� ����
s3_bucket_path = ''
transcribe_json_name = ''
detected_start_times = []

# Loading JSON model
json_file = open('top_models/fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Loading weights
model.load_weights('top_models/fer.h5')

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


@app.route('/')
def index():
    return render_template('index.html')


# ����ڰ� ������ ���ε��ϴ� �κ�
@app.route('/file_upload', methods=['GET', 'POST']) # �����ϴ� url
def file_upload():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)

        try:
            # ������ s3�� ���ε�
            s3 = boto3.client(
                's3',
                aws_access_key_id = aws_access_key_id,
                aws_secret_access_key = aws_secret_access_key
            )
            s3.upload_fileobj(file, aws_bucket_name, filename)
            global s3_bucket_path
            s3_bucket_path = f"s3://{aws_bucket_name}/{filename}"
            response_data = {
                'message': '������ S3 ��Ŷ�� ����Ǿ����ϴ�'
            }
            return json.dumps(response_data), 200, {'Content-Type': 'application/json'}
        except NoCredentialsError:
            response_data = {
                'message': 'AWS �ڰ� ���� ������ ��ȿ���� �ʽ��ϴ�'
            }
            return json.dumps(response_data), 500, {'Content-Type': 'application/json'}
        except Exception as e:
            response_data = {
                'message': str(e)
            }
            return json.dumps(response_data), 500, {'Content-Type': 'application/json'}
    else:
        return render_template('file_upload.html')


@app.route('/do_transcribe', methods=['GET', 'POST'])
def do_transcribe():
    if request.method == 'POST':
        global s3_bucket_path
        if s3_bucket_path == "":
            return "������ ���ε����ּ���"
        global transcribe_json_name
        transcribe_json_name = amazon_transcribe.transcribe_audio(aws_access_key_id, aws_secret_access_key, s3_bucket_path, aws_bucket_name)
        return "���� ��ȯ�� �Ϸ�Ǿ����ϴ�"
    else:
        return render_template('do_transcribe.html')


@app.route('/show_transcribe', methods=['GET', 'POST'])
def show_transcribe():
    global detected_start_times
    global s3_bucket_path
    global transcribe_json_name

    if request.method == 'POST':
        if s3_bucket_path == "":
            return "������ ���ε����ּ���"
        if transcribe_json_name == "":
            return "������ ���ε� ���ּ���"
        result_json_file = show_result.get_json_from_s3(transcribe_json_name, aws_access_key_id, aws_secret_access_key, aws_region_name, aws_bucket_name)
        json_content = json.load(result_json_file)
        detected_start_times, dialogue_save = show_result.extract_dialogue(json_content)
        return render_template('show_transcribe.html', dialogue_save=dialogue_save)
    else:
        # 'dialogue_save' ������ �� ����Ʈ�� �ʱ�ȭ�Ͽ� ��ȯ
        dialogue_save = ([], [])
        return render_template('show_transcribe.html', dialogue_save=dialogue_save)


def perform_emotion_recognition(video_path):
    cap = cv2.VideoCapture(video_path)

    total_emotion_values = np.zeros(8)  # Initialize an array to store total emotion values

    while True:
        ret, img = cap.read()
        if not ret:
            break

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.2, 6)

        for (x, y, w, h) in faces_detected:
            roi_gray = gray_img[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255.0

            predictions = model.predict(img_pixels)
            max_index = int(np.argmax(predictions[0]))

            total_emotion_values += predictions[0]  # Accumulate the emotion values

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Calculate the sum of total emotion values
    total_sum = np.sum(total_emotion_values)

    # Calculate the emotion ratios
    emotion_ratios = total_emotion_values / total_sum

    # Create a dictionary of emotion values
    emotion_values = {}
    emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
    for emotion, ratio in zip(emotions, emotion_ratios):
        emotion_values[emotion] = ratio

    return emotion_values


@app.route('/emotion_recognition', methods=['GET', 'POST'])
def emotion_recognition():
    if request.method == 'POST':
        global s3_bucket_path
        if s3_bucket_path == "":
            return "������ ���ε����ּ���"

        emotion_values = perform_emotion_recognition(s3_bucket_path)
        return render_template('emotion_recognition.html', emotion_values=emotion_values)

    else:
        return render_template('emotion_recognition.html', emotion_values={})


if __name__ == "__main__":
    app.run(debug=True)
    # host ���� ���� �����ϰ� �ʹٸ�
    # app.run(host="127.0.0.1", port="5000", debug=True)