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

# Flask 객체 인스턴스 생성
app = Flask(__name__)

# Amazon S3 정보
aws_access_key_id = os.environ.get("aws_access_key_id")
aws_secret_access_key = os.environ.get("aws_secret_access_key")
aws_region_name = os.environ.get("aws_region_name")
aws_bucket_name = os.environ.get("aws_bucket_name")

# 전역 변수
detected_start_times = []
dialogue_save = [[], []]
emotion_values = {}
speaker_content = []
dialogue_only = []
merged_array = []


@app.route('/')
def index():
    return render_template('index.html')


# !-- 1. 영상 업로드 --!
@app.route('/file_upload', methods=['GET', 'POST'])
def file_upload():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)

        try:
            # 파일을 s3에 업로드
            s3 = boto3.client('s3')
            s3.upload_fileobj(file, aws_bucket_name, filename)
            s3_bucket_path = f"s3://{aws_bucket_name}/{filename}"
            print("파일이 S3 버킷에 저장되었습니다")
            print("bucket path: ", s3_bucket_path)
            response_data = {
                's3_bucket_path': s3_bucket_path,
                'message': '파일이 S3 버킷에 저장되었습니다'
            }
            return json.dumps(response_data), 200, {'Content-Type': 'application/json'}

        except NoCredentialsError:
            print("AWS 자격 증명 정보가 유효하지 않습니다")
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
        return "잘못된 요청입니다. POST 요청을 사용해주세요.", 400


# !-- 2. 음성-텍스트 변환 --!
@app.route('/transcribe', methods=['GET', 'POST'])
def transcribe():
    if request.method == 'POST':
        # JSON 요청 바디에서 's3_bucket_path' 값을 추출
        s3_bucket_path = request.json.get('s3_bucket_path')

        if not s3_bucket_path:
            return "파일을 업로드해주세요"

        transcribe_json_name = transcription.transcription(s3_bucket_path)
        response_data = {
            'transcribe_json_name': transcribe_json_name,
            'message': '음성 변환이 완료되었습니다'
        }
        return json.dumps(response_data), 200, {'Content-Type': 'application/json'}

    else:
        return "잘못된 요청입니다. POST 요청을 사용해주세요.", 400


# !-- 3. 화자 분리 전처리 및 출력, 텍스트 & 표정 감정 분석  --!
@app.route('/diarAndAnalysis', methods=['GET', 'POST'])
def diarAndAnalysis():
    if request.method == 'POST':
        s3_bucket_path = request.json.get('s3_bucket_path')
        transcribe_json_name = request.json.get('transcribe_json_name')

        if s3_bucket_path == "":
            return "파일을 업로드해주세요"
        if transcribe_json_name == "":
            return "파일을 업로드해주세요"

        # 해당 파일 화자분리 및 전처리
        detected_start_times, dialogue_save, speaker_content, dialogue_only, merged_array = speakerDiarization.speakerDiarization(
            transcribe_json_name)
        print(merged_array)  # 연속된 화자의 발언이 있으면 두 발언을 합치는 배열
        print(dialogue_only)  # 내담자의 대화만 저장한 배열

        # speakers = [item for item in dialogue_save[0]]
        # sentences = [item for item in dialogue_save[1]]

        # 표정 감정 분석 수행
        emotion_values = emotion_recognition(s3_bucket_path, detected_start_times)

        # Kobert 서버로 POST 요청
        Koberturl = 'http://3.37.179.243:5000/receive_array'
        headers = {'Content-Type': 'application/json'}
        json_data = json.dumps(dialogue_only)
        response = requests.post(Koberturl, headers=headers, data=json_data)

        # Kobert의 감정분석 결과를 output_json에 저장
        if response.status_code == 200:
            print('transcribe to Kobert 전송 성공')

            output_json = json.loads(response.json()['output_json'])
            sentence_predictions = output_json['predictions']
            total_percentages = output_json['percentages']
            print(sentence_predictions)
            print(total_percentages)
        else:
            print('transcribe to Kobert 전송 실패')

        response_data = {
            'merged_array': merged_array,
            'emotion_values': emotion_values,
            'sentence_predictions': sentence_predictions,
            'total_percentages': total_percentages
        }
        return json.dumps(response_data), 200, {'Content-Type': 'application/json'}
    else:
        return "잘못된 요청입니다. POST 요청을 사용해주세요.", 400


# !-- 4. 표정 감정 인식 --!
def emotion_recognition(s3_bucket_path, detected_start_times):
    global emotion_values

    s3_key = s3_bucket_path.split('/')[-1]  # 추출된 S3 키
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