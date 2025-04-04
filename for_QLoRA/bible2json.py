import csv
import json

# CSV 파일 경로
csv_file_path = "./for_preprocess/Training/pair.csv"
json_file_path = "./for_preprocess/Training/train.json"
# JSON 데이터를 저장할 리스트
json_data = []

# CSV 파일 읽기
with open(csv_file_path, mode="r", encoding="utf-8") as csv_file:
    reader = csv.reader(csv_file)
    next(reader)  # 첫 번째 행(헤더) 건너뛰기

    for row in reader:
        # 각 행을 JSON 형식으로 변환
        message = {
            "messages": [
                {
                    "role": "system",
                    "content": "입력한 글을 개역한글판 성경에 나올법한 어조로 변환하라"
                },
                {
                    "role": "user",
                    "content": row[0]  # 첫 번째 열 (원문)
                },
                {
                    "role": "assistant",
                    "content": row[1]  # 두 번째 열 (결과)
                }
            ]
        }
        json_data.append(message)

# JSON 파일로 저장
with open(json_file_path, mode="w", encoding="utf-8") as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, indent=4)

print(f"JSON 파일이 생성되었습니다: {json_file_path}")
