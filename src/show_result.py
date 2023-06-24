import boto3
import json
import re

def get_json_from_s3(transcribe_json_name, aws_access_key, aws_secret_key, region_name, bucket_name):
    # S3 클라이언트 생성
    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key,
                             region_name=region_name)

    object_key = transcribe_json_name + '.json'

    # 파일 가져오기
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    json_file = response['Body']

    # JSON 파일 반환
    return json_file

def extract_dialogue(json_content):
    target_words = ["지난주는 어떻게", "오늘 상담을 마치"]
    segments = json_content['results']['speaker_labels']['segments']
    items = json_content['results']['items']
    dialogue = []
    detected_start_times = []  # 감지된 start time을 저장할 리스트
    dialogue_save = [[],[]]

    for segment in segments:
        content_list = []
        speaker_label = segment['speaker_label']
        start_time = segment['start_time']

        for item in items:
            if 'start_time' in item and 'end_time' in item and 'alternatives' in item:
                item_start_time = float(item['start_time'])
                item_end_time = float(item['end_time'])

                if item_start_time >= float(segment['start_time']) and item_end_time <= float(segment['end_time']):
                    content_list.append(item['alternatives'][0]['content'])

        dialogue.append({'speaker': speaker_label, 'content': ' '.join(content_list), 'start_time': start_time})

    for line in dialogue:
        speaker = line['speaker']
        content = line['content']
        start_time = float(line['start_time'])

        if len(content) < 10:
            continue

        print(f'{speaker}: {content}')
        dialogue_save[0].append(speaker)  # 첫 번째 하위 배열에 발화자 저장
        dialogue_save[1].append(content)  # 두 번째 하위 배열에 내용 저장

        for target_word in target_words:
            if target_word in content:
                print(f"{speaker}이(가) 시작 시간 {start_time}에 '{target_word}' 단어를 감지했습니다.")
                detected_start_times.append(start_time)  # 감지된 시작 시간을 리스트에 추가

    return detected_start_times, dialogue_save  # 감지된 start time 리스트 반환

