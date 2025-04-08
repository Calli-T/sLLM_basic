# ----- 데이터셋과 모델 준비 -----
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

klue_mrc_dataset = load_dataset('klue', 'mrc', split='train', cache_dir='./for_ignore/datasets_temp')
sentence_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', cache_folder='./for_ignore/model_temp')

# ----- 1000개 선택하고 임베딩을 변환 -----
klue_mrc_dataset = klue_mrc_dataset.train_test_split(train_size=1000, shuffle=False)['train']
embeddings = sentence_model.encode(klue_mrc_dataset['context'])
print(embeddings.shape)

# ----- 라이브러리에 임베딩 저장 -----
import faiss

index = faiss.IndexFlatL2(embeddings.shape[1])  # 인덱스 만들기, 첫 매개변수는 차원수를 지정
index.add(embeddings)  # 인덱스에 임베딩 저장하기

# ----- 의미 검색 하기 -----
query = "이번 연도에는 언제 비가 많이 올까?"
query_embedding = sentence_model.encode([query])
distances, indices = index.search(query_embedding, 3)

for idx in indices[0]:
    print(klue_mrc_dataset['context'][idx][:50])
    # print(klue_mrc_dataset['context'][idx][klue_mrc_dataset[int(idx)]["answers"]["answer_start"][0]:])
    print()
'''# ----- 의미 검색 하기(오답이 나오는 경우) -----
query = klue_mrc_dataset[3]['question']  # 로버트 헨리 딕이 1946년에 매사추세츠 연구소에서 개발한 것은 무엇인가?
query_embedding = sentence_model.encode([query])
distances, indices = index.search(query_embedding, 3)

for idx in indices[0]:
    print(klue_mrc_dataset['context'][idx][:50])
print()
'''
