# - get datasets -
# 기사 데이터
import os
from for_ignore.openai_api_key import get_api_key
from datasets import load_dataset

os.environ["OPENAI_API_KEY"] = get_api_key()

dataset = load_dataset('klue', 'mrc', split='train', cache_dir='./for_ignore/datasets')

# - preprocess -
# 100개 집어 임베딩 벡터로 변환하고 저장
from llama_index.core import Document, VectorStoreIndex

text_list = dataset[:100]["context"]
documents = [Document(text=t) for t in text_list]
index = VectorStoreIndex.from_documents(documents)  # 인덱스 만들기

# - search similar article -
print(dataset[0]["question"])

retrieval_engine = index.as_retriever(similarity_top_k=5, verbose=True)  # 대체 왜 하나 적은 값이 나오는가?
response = retrieval_engine.retrieve(dataset[0]["question"])
print(len(response))  # 5개 맞는데 대체 왜 그런거지? 왜 4개로 나오는거지
print(response[0].node.text)

# - generate with retrieved article -
print(dataset[0]["question"])

query_engine = index.as_query_engine(similarity_top_k=1)
response = query_engine.query(dataset[0]["question"])
print(response)
