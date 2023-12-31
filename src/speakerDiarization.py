import json
import boto3
import os

# Amazon S3 정보
aws_bucket_name = os.environ.get("aws_bucket_name")

def get_json_from_s3(transcribe_json_name):
    # S3 클라이언트 생성
    s3_client = boto3.client('s3')

    object_key = transcribe_json_name + '.json'

    # 파일 가져오기
    response = s3_client.get_object(Bucket = aws_bucket_name, Key=object_key)
    json_file = response['Body']

    # JSON 파일 반환
    return json_file


def speakerDiarization(transcribe_json_name):
    json_file = get_json_from_s3(transcribe_json_name)
    json_content = json.load(json_file)

    target_words = ["지난 주는", "오늘 상담을 마치"] # 첫&마지막 문장 탐지 단어
    try:
        segments = json_content['results']['speaker_labels']['segments']
        items = json_content['results']['items']
    except KeyError:
        print("파일이 올바르게 변환되지 않았습니다. 다른 파일을 시도해 주세요")
        return

    dialogue = []
    dialogue_save = [[],[]]
    speaker_content = []

    detected_start_times = []  # 감지된 start time 저장 리스트
    merged_array = []

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
        speaker_content.append([speaker, content])

        merged_array = []

        previous_speaker = None

        for item in speaker_content:
            speaker = item[0]
            message = item[1]

            if previous_speaker is None or previous_speaker != speaker:
                merged_array.append([speaker, message])
            else:
                merged_array[-1][1] += ' ' + message

            previous_speaker = speaker


        for target_word in target_words:
            if target_word in content:
                print(f"{speaker}이(가) 시작 시간 {start_time}에 '{target_word}' 단어를 감지했습니다.")
                detected_start_times.append(start_time)  # 감지된 시작 시간을 리스트에 추가

    speaker1_array = [item for item in merged_array if item[0] == 'spk_1']
    dialogue_only = [item[1] for item in speaker1_array]

    print(dialogue_save)
    return detected_start_times, dialogue_save, speaker_content, dialogue_only, merged_array  # 감지된 start time 리스트 반환