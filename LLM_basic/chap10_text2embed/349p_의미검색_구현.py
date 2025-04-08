from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# 데이터셋과 모델 준비
klue_mrc_dataset = load_dataset('klue', 'mrc', split='train', cache_dir='./datasets_temp')
sentence_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', cache_folder='./model_temp')

# 1000개 선택하고 임베딩을 변환
klue_mrc_dataset = klue_mrc_dataset.train_test_split(train_size=1000, shuffle=False)['train']
embeddings = sentence_model.encode(klue_mrc_dataset['context'])
print(embeddings.shape)

import faiss
# 인덱스 만들기
index = faiss.IndexFlatL2(embeddings.shape[1])
# 인덱스에 임베딩 저장하기
index.add(embeddings)





