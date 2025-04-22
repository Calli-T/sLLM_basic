'''
pinecone의 CRUD를 실습해보자 1
인덱스 생성 -> 임베딩 생성 -> 전처리 -> 업로드까지
'''
# - 파인콘 계정 연결과 인덱스(스키마 정도?) 생성 -
from pinecone import Pinecone
from for_ignore.pinecone_api_key import get_api_key

pinecone_api_key = get_api_key()
pc = Pinecone(api_key=pinecone_api_key)

# 인덱스 생성과 가져오기, 아마존 리전은 대체 뭔 소리냐?
# 생성을 두 번 할 필요는 없지
# pc.create_index("llm-book", spec=ServerlessSpec("aws", "us-east-1"), dimension=768)
index = pc.Index("llm-book")

# - 임베딩 생성 -
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS", cache_folder='./for_ignore/embedding_model')
klue_dp_train = load_dataset('klue', 'dp', split='train[:100]', cache_dir='for_ignore/dataset_klue_dp')
embeddings = sentence_model.encode(klue_dp_train['sentence'])

# - 전처리 for 파인콘 입력 -
# 파이썬 기본 데이터 타입으로 변경과 필요한 자료만 남기기
# 파이썬 기본 데이터 타입으로 변경
embeddings = embeddings.tolist()
# {"id": 문서 ID(str), "values": 벡터 임베딩(List[float]), "metadata": 메타 데이터(dict) ) 형태로 데이터 준비
insert_data = []
for idx, (embedding, text) in enumerate(zip(embeddings, klue_dp_train['sentence'])):
    insert_data.append({"id": str(idx), "values": embedding, "metadata": {'text': text}})

# - 생성해둔 파인콘 인덱스에 임베딩을 저장 -
# 네임스페이스는 테이블쯤 되려나?
upsert_response = index.upsert(vectors=insert_data, namespace='llm-book-sub')
