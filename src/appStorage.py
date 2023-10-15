# -*- coding: cp949 -*-
import json
import boto3
import requests
from botocore.exceptions import NoCredentialsError
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import perform_emotion_recognition
import speakerDiarization, transcription
import os

# Flask ��ü �ν��Ͻ� ����
app = Flask(__name__)

# Amazon S3 ����
aws_access_key_id = os.environ.get("aws_access_key_id")
aws_secret_access_key = os.environ.get("aws_secret_access_key")
aws_region_name = os.environ.get("aws_region_name")
aws_bucket_name = os.environ.get("aws_bucket_name")

# ���� ����
detected_start_times = []
dialogue_save = [[], []]
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
        return "�߸��� ��û�Դϴ�. POST ��û�� ������ּ���.", 400


# !-- 2. ����-�ؽ�Ʈ ��ȯ --!
@app.route('/transcribe', methods=['GET', 'POST'])
def transcribe():
    if request.method == 'POST':
        # JSON ��û �ٵ𿡼� 's3_bucket_path' ���� ����
        s3_bucket_path = request.json.get('s3_bucket_path')

        if not s3_bucket_path:
            return "������ ���ε����ּ���"

        transcribe_json_name = transcription.transcription(s3_bucket_path)
        response_data = {
            'transcribe_json_name': transcribe_json_name,
            'message': '���� ��ȯ�� �Ϸ�Ǿ����ϴ�'
        }
        return json.dumps(response_data), 200, {'Content-Type': 'application/json'}

    else:
        return "�߸��� ��û�Դϴ�. POST ��û�� ������ּ���.", 400


# !-- 3. ȭ�� �и� ��ó�� �� ���, �ؽ�Ʈ & ǥ�� ���� �м�  --!
@app.route('/diarAndAnalysis', methods=['GET', 'POST'])
def diarAndAnalysis():
    if request.method == 'POST':
        s3_bucket_path = request.json.get('s3_bucket_path')
        transcribe_json_name = request.json.get('transcribe_json_name')

        if s3_bucket_path == "":
            return "������ ���ε����ּ���"
        if transcribe_json_name == "":
            return "������ ���ε����ּ���"

        # �ش� ���� ȭ�ںи� �� ��ó��
        detected_start_times, dialogue_save, speaker_content, dialogue_only, merged_array = speakerDiarization.speakerDiarization(
            transcribe_json_name)
        print(merged_array)  # ���ӵ� ȭ���� �߾��� ������ �� �߾��� ��ġ�� �迭
        print(dialogue_only)  # �������� ��ȭ�� ������ �迭

        # speakers = [item for item in dialogue_save[0]]
        # sentences = [item for item in dialogue_save[1]]

        # ǥ�� ���� �м� ����
        emotion_values = emotion_recognition(s3_bucket_path, detected_start_times)

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
            'merged_array': merged_array,
            'emotion_values': emotion_values,
            'sentence_predictions': sentence_predictions,
            'total_percentages': total_percentages
        }
        return json.dumps(response_data), 200, {'Content-Type': 'application/json'}
    else:
        return "�߸��� ��û�Դϴ�. POST ��û�� ������ּ���.", 400


# !-- 4. ǥ�� ���� �ν� --!
def emotion_recognition(s3_bucket_path, detected_start_times):
    global emotion_values

    s3_key = s3_bucket_path.split('/')[-1]  # ����� S3 Ű
    local_file_path = 'video.mp4'

    perform_emotion_recognition.download_file_from_s3(s3_key, local_file_path)
    emotion_values = perform_emotion_recognition.perform_emotion_recognition(local_file_path, detected_start_times)

    return emotion_values


#
# @app.route("/convey")
# def convey():
#     data = {
#         "emotion_values": emotion_values,
#         "merged_array": merged_array
#     }
#     return json.dumps(data)


if __name__ == "__main__":
    app.run(debug=True)