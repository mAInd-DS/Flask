# -*- coding: cp949 -*- 

import json
import boto3
import requests
from botocore.exceptions import NoCredentialsError
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import show_result, amazon_transcribe, perform_emotion_recognition
from dotenv import load_dotenv
import cv2
import numpy as np
from keras.models import model_from_json
from keras.utils.image_utils import img_to_array
import show_result, amazon_transcribe
import os

# Flask 객체 인스턴스 생성
app = Flask(__name__)

# Amazon S3 정보 - 보안 유지 요망
aws_access_key_id = os.environ.get("aws_access_key_id")
aws_secret_access_key = os.environ.get("aws_secret_access_key")
aws_region_name = os.environ.get("aws_region_name")
aws_bucket_name = os.environ.get("aws_bucket_name")


# 전역 변수
s3_bucket_path = ''
transcribe_json_name = ''
detected_start_times = []
dialogue_save = [[],[]]
emotion_values = {}
speaker_content = []

@app.route('/')
def index():
    return render_template('index.html')


# 사용자가 영상을 업로드하는 부분
@app.route('/file_upload', methods=['GET', 'POST']) # 접속하는 url
def file_upload():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)

        try:
            # 파일을 s3에 업로드
            s3 = boto3.client(
                's3',
                aws_access_key_id = aws_access_key_id,
                aws_secret_access_key = aws_secret_access_key
            )
            s3.upload_fileobj(file, aws_bucket_name, filename)
            global s3_bucket_path
            s3_bucket_path = f"s3://{aws_bucket_name}/{filename}"
            response_data = {
                'message': '파일이 S3 버킷에 저장되었습니다'
            }
            return json.dumps(response_data), 200, {'Content-Type': 'application/json'}
        except NoCredentialsError:
            response_data = {
                'message': 'AWS 자격 증명 정보가 유효하지 않습니다'
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
            return "파일을 업로드해주세요"
        global transcribe_json_name
        transcribe_json_name = amazon_transcribe.transcribe_audio(aws_access_key_id, aws_secret_access_key, s3_bucket_path, aws_bucket_name)
        return "음성 변환이 완료되었습니다"
    else:
        return render_template('do_transcribe.html')

@app.route('/show_transcribe', methods=['GET', 'POST'])
def show_transcribe():
    global detected_start_times
    global s3_bucket_path
    global transcribe_json_name
    global dialogue_save
    global speaker_content

    if request.method == 'POST':
        if s3_bucket_path == "":
            return "파일을 업로드해주세요"
        if transcribe_json_name == "":
            return "파일을 업로드 해주세요"

        result_json_file = show_result.get_json_from_s3(transcribe_json_name, aws_access_key_id, aws_secret_access_key, aws_region_name, aws_bucket_name)
        json_content = json.load(result_json_file)
        detected_start_times, dialogue_save, speaker_content = show_result.extract_dialogue(json_content)

        url = 'http://127.0.0.1:5000/receive_transcribe'
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, json=dialogue_save)
        if response.status_code == 200:
            print('transcribe 전송 성공')
        else:
            print('transcribe 전송 실패')
        print(dialogue_save)
        print(speaker_content)
        return render_template('show_transcribe.html', dialogue_save=dialogue_save)
    else:
        # 'dialogue_save' 변수를 빈 리스트로 초기화하여 반환
        dialogue_save = ([], [])
        return render_template('show_transcribe.html', dialogue_save=dialogue_save)

#
# @app.route('/send_transcribe', methods=['GET', 'POST'])
# def send_transcribe():
#     global dialogue_save
#     if request.method == 'POST':
#         # 다른 Flask 서버로 배열 전송 (가정)
#         url = 'http://127.0.0.1:5002/receive_transcribe'
#         headers = {'Content-Type': 'application/json'}
#         payload = json.dumps(dialogue_save)
#         response = requests.post(url, headers=headers, data=payload)
#         if response.status_code == 200:
#             print('transcribe 전송 성공')
#         else:
#             print('transcribe 전송 실패')


@app.route('/emotion_recognition', methods=['GET', 'POST'])
def emotion_recognition():
    global emotion_values
    if request.method == 'POST':
        global s3_bucket_path
        if s3_bucket_path == "":
            return "파일을 업로드해주세요"

        s3_key = s3_bucket_path.split('/')[-1]  # 추출된 S3 키
        local_file_path = 'video.mp4'

        perform_emotion_recognition.download_file_from_s3(aws_bucket_name, s3_key, local_file_path, aws_access_key_id, aws_secret_access_key)

        emotion_values = perform_emotion_recognition.perform_emotion_recognition(local_file_path, detected_start_times)
        print(emotion_values)
        return render_template('emotion_recognition.html', emotion_values=emotion_values)
    else:
        return render_template('emotion_recognition.html', emotion_values={})

@app.route("/convey")
def convey():
    return emotion_values

if __name__ == "__main__":
    app.run(debug=True)
    # host 등을 직접 지정하고 싶다면
    # app.run(host="127.0.0.1", port="5000", debug=True)