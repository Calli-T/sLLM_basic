# list를 살리고, 최상위에 대괄호를 둔 채로 json을 jsonl로 변경

import json

input_file = "/mnt/additional/projects/sLLM/for_QLoRA/for_preprocess/Training/train.json"
output_file = "/mnt/additional/projects/sLLM/for_QLoRA/for_preprocess/Training/train.jsonl"

# JSON 파일 읽기
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)  # 리스트 형태로 로드

# JSONL 스타일로 변환 및 저장
with open(output_file, "w", encoding="utf-8") as f:
    for obj in data:
        f.write(json.dumps(obj["messages"], ensure_ascii=False) + "\n")  # 대괄호 유지

print(f"변환 완료! JSONL 파일이 '{output_file}'로 저장되었습니다.")
