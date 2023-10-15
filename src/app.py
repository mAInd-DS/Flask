# -*- coding: cp949 -*-
import json
import boto3
import requests
from botocore.exceptions import NoCredentialsError
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import perform_emotion_recognition
import show_result, amazon_transcribe
import os

# Flask ��ü �ν��Ͻ� ����
app = Flask(__name__)

# Amazon S3 ����
aws_access_key_id = os.environ.get("aws_access_key_id")
aws_secret_access_key = os.environ.get("aws_secret_access_key")
aws_region_name = os.environ.get("aws_region_name")
aws_bucket_name = os.environ.get("aws_bucket_name")

# ���� ����
s3_bucket_path = ''
transcribe_json_name = ''
detected_start_times = []
dialogue_save = [[],[]]
emotion_values = {}
speaker_content = []
dialogue_only = []
merged_array = []


@app.route('/')
def index():
    return render_template('index.html')


# !-- 1. ���� ���ε� --!
@app.route('/file_upload', methods=['GET', 'POST'])
def file_upload():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)

        try:
            # ������ s3�� ���ε�
            s3 = boto3.client('s3')
            s3.upload_fileobj(file, aws_bucket_name, filename)
            global s3_bucket_path
            s3_bucket_path = f"s3://{aws_bucket_name}/{filename}"
            print("������ S3 ��Ŷ�� ����Ǿ����ϴ�")
            response_data = {
                'message': '������ S3 ��Ŷ�� ����Ǿ����ϴ�'
            }
            return json.dumps(response_data), 200, {'Content-Type': 'application/json'}
        except NoCredentialsError:
            print("AWS �ڰ� ���� ������ ��ȿ���� �ʽ��ϴ�")
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


# !-- 2. ����-�ؽ�Ʈ ��ȯ --!
@app.route('/do_transcribe', methods=['GET', 'POST'])
def do_transcribe():
    if request.method == 'POST':
        global s3_bucket_path
        if s3_bucket_path == "":
            return "������ ���ε����ּ���"
        global transcribe_json_name
        transcribe_json_name = amazon_transcribe.transcribe_audio(s3_bucket_path)
        return "���� ��ȯ�� �Ϸ�Ǿ����ϴ�"
    else:
        return render_template('do_transcribe.html')


# !-- 3. ȭ�� �и� ��ó�� �� ���, ���� �м� ��û --!
@app.route('/show_transcribe', methods=['GET', 'POST'])
def show_transcribe():
    global detected_start_times # Ư���ܾ� �����ð�
    global s3_bucket_path
    global transcribe_json_name
    global dialogue_save # html�� ���޵Ǵ� ��ȭ��ġ
    global speaker_content
    global dialogue_only
    global merged_array

    if request.method == 'POST':
        if s3_bucket_path == "":
            return "������ ���ε����ּ���"
        if transcribe_json_name == "":
            return "������ ���ε� ���ּ���"

        # s3 ��Ŷ���� amazon transcribe json ���� ��������
        result_json_file = show_result.get_json_from_s3(transcribe_json_name)
        json_content = json.load(result_json_file)

        # �ش� ���� ȭ�ںи� �� ��ó��
        detected_start_times, dialogue_save, speaker_content, dialogue_only, merged_array = show_result.extract_dialogue(json_content)
        print(merged_array) # ���ӵ� ȭ���� �߾��� ������ �� �߾��� ��ġ�� �迭
        print(dialogue_only) # �������� ��ȭ�� ������ �迭

        # Kobert ������ POST ��û
        detected_start_times, dialogue_save, speaker_content, dialogue_only, merged_array = show_result.extract_dialogue(json_content)
        print(merged_array)
        print(dialogue_only)
        #
        # speakers = [item for item in dialogue_save[0]]
        # sentences = [item for item in dialogue_save[1]]

        url = 'http://3.37.179.243:5000/receive_array'
        headers = {'Content-Type': 'application/json'}
        json_data = json.dumps(dialogue_only)
        response = requests.post(url, headers=headers, data=json_data)
        if response.status_code == 200:
            print('transcribe ���� ����')
        else:
            print('transcribe ���� ����')

        print(dialogue_save)
        print(speaker_content)
        print(dialogue_only)
        return render_template('show_transcribe.html', dialogue_save=dialogue_save)
    else:
        # 'dialogue_save' ������ �� ����Ʈ�� �ʱ�ȭ�Ͽ� ��ȯ
        dialogue_save = ([], [])
        return render_template('show_transcribe.html', dialogue_save=dialogue_save)


# !-- 4. ǥ�� ���� �ν� --!
@app.route('/emotion_recognition', methods=['GET', 'POST'])
def emotion_recognition():
    global emotion_values
    if request.method == 'POST':
        global s3_bucket_path
        if s3_bucket_path == "":
            return "������ ���ε����ּ���"

        s3_key = s3_bucket_path.split('/')[-1]  # ����� S3 Ű
        local_file_path = 'video.mp4'

        perform_emotion_recognition.download_file_from_s3(s3_key, local_file_path)

        emotion_values = perform_emotion_recognition.perform_emotion_recognition(local_file_path, detected_start_times)
        print(emotion_values)
        return render_template('emotion_recognition.html', emotion_values=emotion_values)
    else:
        return render_template('emotion_recognition.html', emotion_values={})


# �ؽ�Ʈ �����м� ��� json ���� ������ ���
@app.route('/get_json', methods=['GET'])
def get_json():
    url = 'http://maind-meeting.shop:5000/show'  # JSON ������ �ִ� URL
    response = requests.get(url)
    json_data = response.json()
    json_data_str = json.dumps(json_data, ensure_ascii = False)
    print(json_data_str)
    return json_data_str


@app.route("/convey")
def convey():
    data = {
        "emotion_values": emotion_values,
        "merged_array": merged_array
    }
    return json.dumps(data)
    

if __name__ == "__main__":
    app.run(debug=True)