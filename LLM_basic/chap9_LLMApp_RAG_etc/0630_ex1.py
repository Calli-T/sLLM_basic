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


def get_llama_index_base():
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


def get_inner_process():
    # ----- -----
    from llama_index.core import (
        VectorStoreIndex,
        get_response_synthesizer,
    )
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.postprocessor import SimilarityPostprocessor

    # 검색을 위한 retriever 생성
    retriever = VectorIndexRetriever(index=index, similarity_top_k=1)

    # 검색 결과를 질문과 결합하는 synthesizer
    response_synthesizer = get_response_synthesizer()

    # 두 요소를 결합한 쿼리 엔진
    query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer,
                                        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]) # 기본값 gpt 3.5 turbo
    # 유사도는 0.7초과만 필터린

    # RAG 수행
    # response = query_engine.query("북태평양 기단은 어디와 상호작용하는가?")
    response = query_engine.query("장마 기간은 얼마나 될까?")
    # response = query_engine.query("북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?")
    print(response)

# 얘 잘 안돌아가는데?
# get_inner_process()