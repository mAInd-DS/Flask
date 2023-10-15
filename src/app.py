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

# Flask 객체 인스턴스 생성
app = Flask(__name__)

# Amazon S3 정보
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
            global s3_bucket_path
            s3_bucket_path = f"s3://{aws_bucket_name}/{filename}"
            print("파일이 S3 버킷에 저장되었습니다")
            response_data = {
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
        return render_template('file_upload.html')


# !-- 2. 음성-텍스트 변환 --!
@app.route('/do_transcribe', methods=['GET', 'POST'])
def do_transcribe():
    if request.method == 'POST':
        global s3_bucket_path
        if s3_bucket_path == "":
            return "파일을 업로드해주세요"
        global transcribe_json_name
        transcribe_json_name = amazon_transcribe.transcribe_audio(s3_bucket_path)
        return "음성 변환이 완료되었습니다"
    else:
        return render_template('do_transcribe.html')


# !-- 3. 화자 분리 전처리 및 출력, 감정 분석 요청 --!
@app.route('/show_transcribe', methods=['GET', 'POST'])
def show_transcribe():
    global detected_start_times # 특정단어 감지시간
    global s3_bucket_path
    global transcribe_json_name
    global dialogue_save # html에 전달되는 대화뭉치
    global speaker_content
    global dialogue_only
    global merged_array

    if request.method == 'POST':
        if s3_bucket_path == "":
            return "파일을 업로드해주세요"
        if transcribe_json_name == "":
            return "파일을 업로드 해주세요"

        # s3 버킷에서 amazon transcribe json 파일 가져오기
        result_json_file = show_result.get_json_from_s3(transcribe_json_name)
        json_content = json.load(result_json_file)

        # 해당 파일 화자분리 및 전처리
        detected_start_times, dialogue_save, speaker_content, dialogue_only, merged_array = show_result.extract_dialogue(json_content)
        print(merged_array) # 연속된 화자의 발언이 있으면 두 발언을 합치는 배열
        print(dialogue_only) # 내담자의 대화만 저장한 배열

        # Kobert 서버로 POST 요청
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
            print('transcribe 전송 성공')
        else:
            print('transcribe 전송 실패')

        print(dialogue_save)
        print(speaker_content)
        print(dialogue_only)
        return render_template('show_transcribe.html', dialogue_save=dialogue_save)
    else:
        # 'dialogue_save' 변수를 빈 리스트로 초기화하여 반환
        dialogue_save = ([], [])
        return render_template('show_transcribe.html', dialogue_save=dialogue_save)


# !-- 4. 표정 감정 인식 --!
@app.route('/emotion_recognition', methods=['GET', 'POST'])
def emotion_recognition():
    global emotion_values
    if request.method == 'POST':
        global s3_bucket_path
        if s3_bucket_path == "":
            return "파일을 업로드해주세요"

        s3_key = s3_bucket_path.split('/')[-1]  # 추출된 S3 키
        local_file_path = 'video.mp4'

        perform_emotion_recognition.download_file_from_s3(s3_key, local_file_path)

        emotion_values = perform_emotion_recognition.perform_emotion_recognition(local_file_path, detected_start_times)
        print(emotion_values)
        return render_template('emotion_recognition.html', emotion_values=emotion_values)
    else:
        return render_template('emotion_recognition.html', emotion_values={})


# 텍스트 감정분석 결과 json 파일 얻어오는 경로
@app.route('/get_json', methods=['GET'])
def get_json():
    url = 'http://maind-meeting.shop:5000/show'  # JSON 파일이 있는 URL
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