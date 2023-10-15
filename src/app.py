# -*- coding: cp949 -*-
import json
import boto3
import requests
from botocore.exceptions import NoCredentialsError
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import perform_emotion_recognition
import speakerDiarization, transcribe
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
            print("bucket path: ", s3_bucket_path)
            response_data = {
                's3_bucket_path': s3_bucket_path,
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
        # JSON ��û �ٵ𿡼� 's3_bucket_path' ���� ����
        s3_bucket_path = request.json.get('s3_bucket_path')

        if not s3_bucket_path:
            return "������ ���ε����ּ���"

        transcribe_json_name = transcribe.transcribe(s3_bucket_path)
        response_data = {
            'transcribe_json_name': transcribe_json_name,
            'message': '���� ��ȯ�� �Ϸ�Ǿ����ϴ�'
        }
        return json.dumps(response_data), 200, {'Content-Type': 'application/json'}

    else:
        return render_template('do_transcribe.html')


# !-- 3. ȭ�� �и� ��ó�� �� ���, ���� �м� ��û --!
@app.route('/show_transcribe', methods=['GET', 'POST'])
def show_transcribe():

    global output_json
    if request.method == 'POST':
        s3_bucket_path = request.json.get('s3_bucket_path')
        transcribe_json_name = request.json.get('transcribe_json_name')

        if s3_bucket_path == "":
            return "������ ���ε����ּ���"
        if transcribe_json_name == "":
            return "������ ���ε����ּ���"

        # �ش� ���� ȭ�ںи� �� ��ó��
        detected_start_times, dialogue_save, speaker_content, dialogue_only, merged_array = speakerDiarization.speakerDiarization(transcribe_json_name)
        print(merged_array) # ���ӵ� ȭ���� �߾��� ������ �� �߾��� ��ġ�� �迭
        print(dialogue_only) # �������� ��ȭ�� ������ �迭

        # speakers = [item for item in dialogue_save[0]]
        # sentences = [item for item in dialogue_save[1]]


        # Kobert ������ POST ��û
        Koberturl = 'http://3.37.179.243:5000/receive_array'
        headers = {'Content-Type': 'application/json'}
        json_data = json.dumps(dialogue_only)
        response = requests.post(Koberturl, headers=headers, data=json_data)

        # Kobert�� �����м� ����� output_json�� ����
        if response.status_code == 200:
            print('transcribe to Kobert ���� ����')

            output_json = json.loads(response.json()['output_json'])
            sentence_predictions = output_json['predictions']
            total_percentages = output_json['percentages']
            print(sentence_predictions)
            print(total_percentages)
        else:
            print('transcribe to Kobert ���� ����')


        response_data = {
            'detected_start_times': detected_start_times,
            'dialogue': dialogue_save,
            'merged_array': merged_array,
            'sentence_predictions': sentence_predictions,
            'total_percentages': total_percentages
        }
        return json.dumps(response_data), 200, {'Content-Type': 'application/json'}
    else:
        # 'dialogue_save' ������ �� ����Ʈ�� �ʱ�ȭ�Ͽ� ��ȯ
        dialogue_save = ([], [])
        return render_template('show_transcribe.html', dialogue_save=dialogue_save)


# !-- 4. ǥ�� ���� �ν� --!
@app.route('/emotion_recognition', methods=['GET', 'POST'])
def emotion_recognition():
    global emotion_values
    if request.method == 'POST':
        s3_bucket_path = request.json.get('s3_bucket_path')
        detected_start_times = request.json.get('detected_start_times')
        if s3_bucket_path == "":
            return "������ ���ε����ּ���"

        s3_key = s3_bucket_path.split('/')[-1]  # ����� S3 Ű
        local_file_path = 'video.mp4'
        perform_emotion_recognition.download_file_from_s3(s3_key, local_file_path)
        emotion_values = perform_emotion_recognition.perform_emotion_recognition(local_file_path, detected_start_times)

        response_data = {
            'emotion_values': emotion_values
        }
        return json.dumps(response_data), 200, {'Content-Type': 'application/json'}
    else:
        return render_template('emotion_recognition.html', emotion_values={})


# # �ؽ�Ʈ �����м� ��� json ���� ������ ���
# @app.route('/get_json', methods=['GET'])
# def get_json():
#     url = 'http://maind-meeting.shop:5000/show'  # JSON ������ �ִ� URL
#     response = requests.get(url)
#     json_data = response.json()
#     json_data_str = json.dumps(json_data, ensure_ascii = False)
#     print(json_data_str)
#     return json_data_str

@app.route("/convey")
def convey():
    data = {
        "emotion_values": emotion_values,
        "merged_array": merged_array
    }
    return json.dumps(data)
    

if __name__ == "__main__":
    app.run(debug=True)