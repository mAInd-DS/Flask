from datetime import datetime
import time
import boto3

def transcribe_audio(aws_access_key_id, aws_secret_access_key, audio_file_uri, bucket_name):

    # Transcribe 클라이언트 생성
    transcribe = boto3.client(
        'transcribe',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name='ap-northeast-2'
    )

    media = {'MediaFileUri': audio_file_uri}
    settings = {
        'ShowSpeakerLabels': True,
        'MaxSpeakerLabels': 2
    }

    # Transcribe 작업 시작
    print("transcribe 작업을 시작합니다")
    job_name = 'transcribe-job-' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media=media,
        MediaFormat='mp4',  # 음원 파일 형식에 맞게 변경
        LanguageCode='ko-KR',
        OutputBucketName=bucket_name,  # 결과 파일을 업로드할 버킷 이름 추가
        Settings = settings
    )

    # Transcription 작업 완료 대기
    while True:
        response = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        job_status = response['TranscriptionJob']['TranscriptionJobStatus']

        if job_status == 'COMPLETED':
            print("Transcription COMPLETED! job name: "+job_name)
            return job_name

        if job_status == 'FAILED':
            print("Transcription FAILED!")
            break
        else:
            print("Transcription in progress. Current status: " + job_status)
            time.sleep(10)  # 10초 대기 후 다시 확인




