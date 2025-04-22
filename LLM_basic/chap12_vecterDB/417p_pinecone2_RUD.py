'''
pinecone의 CRUD를 실습해보자 2
파인콘 어뎁터 결합 -> 인덱스 가져오기 -> retrieve & update & delete
'''
# 파인콘 클라 & 인덱스 지정
from pinecone import Pinecone, ServerlessSpec
from for_ignore.pinecone_api_key import get_api_key

pinecone_api_key = get_api_key()
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("llm-book")

# - 임베딩 생성과 전처리 -
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS", cache_folder='./for_ignore/embedding_model')
klue_dp_train = load_dataset('klue', 'dp', split='train[:100]', cache_dir='for_ignore/dataset_klue_dp')
embeddings = sentence_model.encode(klue_dp_train['sentence'])
embeddings = embeddings.tolist()
insert_data = []
for idx, (embedding, text) in enumerate(zip(embeddings, klue_dp_train['sentence'])):
    insert_data.append({"id": str(idx), "values": embedding, "metadata": {'text': text}})

# - 검색 -
query_response = index.query(
    namespace='llm-book-sub',  # 검색할 네임스페이스
    top_k=10,  # 몇 개의 결과를 반환할지
    include_values=True,  # 벡터 임베딩 반환 여부
    include_metadata=True,  # 메타 데이터 반환 여부
    vector=embeddings[0]  # 검색할 벡터 임베딩
)
print(query_response)

# - 수정 및 삭제 -
new_text = '얄리 얄리 얄라셩 얄라리 얄라'
new_embedding = sentence_model.encode(new_text).tolist()
# 업데이트
update_response = index.update(
    id='1',
    values=new_embedding,
    set_metadata={'text': new_text},
    namespace='llm-book-sub'
)

# 삭제
delete_response = index.delete(ids=['79'], namespace='llm-book-sub')
