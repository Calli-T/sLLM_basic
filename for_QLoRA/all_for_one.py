import os
import json


def list_json_files(directory):
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def process_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    passage = data["Meta(Refine)"]["passage"].replace("\n", " ")
    summaries = " ".join([data["Annotation"].get(key, "") for key in ["summary1", "summary2", "summary3"] if
                          data["Annotation"].get(key)])

    return {"messages": [
        {"role": "system", "content": "다음을 요약하세요."},
        {"role": "user", "content": passage},
        {"role": "assistant", "content": summaries}
    ]}


def process_and_save_json(input_files, output_file):
    result = []
    for file_path in input_files:
        result.append(process_json_file(file_path))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


# 사용 예시
directory_path = "./for_preprocess/Training"  # 검색할 디렉터리 경로를 지정
json_file_list = list_json_files(directory_path)
output_path = "for_preprocess/Training/train.json"  # 결과 파일 저장 경로
process_and_save_json(json_file_list, output_path)
