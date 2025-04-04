import csv

# CSV 파일 경로
csv_file_path = "./for_preprocess/Training/pair.csv"

# 최대 단어 수를 저장할 변수
max_word_count = 0

# CSV 파일 읽기
with open(csv_file_path, mode="r", encoding="utf-8") as csv_file:
    reader = csv.reader(csv_file)
    next(reader)  # 첫 번째 행(헤더) 건너뛰기

    for row in reader:
        # 각 행의 첫 번째와 두 번째 문장에 대해 단어 수 계산
        first_sentence_word_count = len(row[0].split())  # 첫 번째 문장 단어 수
        second_sentence_word_count = len(row[1].split())  # 두 번째 문장 단어 수

        # 둘 중 더 큰 단어 수를 업데이트
        max_word_count = max(max_word_count, first_sentence_word_count + second_sentence_word_count)

# 결과 출력
print(f"가장 많은 단어를 포함한 문장의 단어 수: {max_word_count}")
