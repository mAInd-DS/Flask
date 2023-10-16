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
        survey_id_json = request.form.get('survey_id')
        survey_id_data = json.loads(survey_id_json)
        survey_id = survey_id_data.get('survey_id')

        file = request.files['file']
        filename = secure_filename(file.filename)

        try:
            # ������ s3�� ���ε�
            s3 = boto3.client('s3')
            s3.upload_fileobj(file, aws_bucket_name, filename)
            s3_bucket_path = f"s3://{aws_bucket_name}/{filename}"
            print("bucket path: ", s3_bucket_path)

            transcribe_json_name = transcribe(s3_bucket_path)
            merged_array, emotion_values, sentence_predictions, total_percentages = diarAndAnalysis(s3_bucket_path, transcribe_json_name)

            response_data = {
                'survey_id': survey_id,
                'merged_array': merged_array,
                'emotion_values': emotion_values,
                'sentence_predictions': sentence_predictions,
                'total_percentages': total_percentages
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


# # !-- 2. ����-�ؽ�Ʈ ��ȯ --!
def transcribe(s3_bucket_path):
    transcribe_json_name = transcription.transcription(s3_bucket_path)
    print('���� ��ȯ�� �Ϸ�Ǿ����ϴ�')
    return transcribe_json_name


# !-- 3. ȭ�� �и� ��ó�� �� ���, �ؽ�Ʈ & ǥ�� ���� �м�  --!
def diarAndAnalysis(s3_bucket_path, transcribe_json_name):
    # 1. ȭ�ںи� �� ��ó��
    detected_start_times, dialogue_save, speaker_content, dialogue_only, merged_array = speakerDiarization.speakerDiarization(
        transcribe_json_name)
    print(merged_array)  # ���ӵ� ȭ���� �߾��� ������ �� �߾��� ��ġ�� �迭

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

    return merged_array, emotion_values, sentence_predictions, total_percentages


# !-- 4. ǥ�� ���� �м�  --!
def emotion_recognition(s3_bucket_path, detected_start_times):
    global emotion_values

    s3_key = s3_bucket_path.split('/')[-1]  # ����� S3 Ű
    local_file_path = 'video.mp4'

    perform_emotion_recognition.download_file_from_s3(s3_key, local_file_path)
    emotion_values = perform_emotion_recognition.perform_emotion_recognition(local_file_path, detected_start_times)

    return emotion_values


if __name__ == "__main__":
    app.run(debug=True)