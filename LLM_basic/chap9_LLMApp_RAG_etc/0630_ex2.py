# LLM 캐시 실습
# 유사 캐시
import os
import time

import chromadb
from openai import OpenAI

from for_ignore.openai_api_key import get_api_key

os.environ["OPENAI_API_KEY"] = get_api_key()

openai_client = OpenAI()
chroma_client = chromadb.Client()


# 클래스가 없을 때 걸리는 시간
def timelog_without_cache():
    def response_text(openai_resp):
        return openai_resp.choices[0].message.content

    question = "북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?"
    for _ in range(2):
        start_time = time.time()
        response = openai_client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {
                    'role': 'user',
                    'content': question
                }
            ],
        )
        response = response_text(response)
        print(f'질문: {question}')
        print("소요 시간: {:.2f}s".format(time.time() - start_time))
        print(f'답변: {response}\n')


# timelog_without_cache()

# 일치캐시 클래스(파이썬 딕셔너리)
class OpenAICache:
    def __init__(self, _openai_client):
        self.openai_client = _openai_client
        self.cache = {}

    def response_text(self, openai_resp):
        return openai_resp.choices[0].message.content

    def generate(self, prompt):
        if prompt not in self.cache:  # 프롬프트가 캐시에 없으면 보내서 답을 받고, 캐시에 프롬프트와 답(응답의 텍스트 부분)을 쌍으로 저장
            response = self.openai_client.chat.completions.create(model='gpt-3.5-turbo',
                                                                  messages=[{
                                                                      'role': 'user',
                                                                      'content': prompt
                                                                  }],
                                                                  )
            self.cache[prompt] = self.response_text(response)
        return self.cache[prompt]  # 이게 있다면, 따로 llm을 굴리거나 저장할 필요도 없을 것


# - run -
def same_cache():
    openai_cache = OpenAICache(openai_client)

    question = '북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?'
    for _ in range(2):
        start_time = time.time()
        response = openai_cache.generate(question)
        print(f'질문: {question}')
        print("소요 시간: {:.2f}s".format(time.time() - start_time))
        print(f'답변: {response}\n')


# same_cache()

# 유사 검색 캐시 구현 클래스, 313p
# 벡터 db 클라이언트인, semantic_cache를 init에 줘야한다
## 엄밀히 따지면, 유사 캐시말고 일치 캐시도 같이 사용함

class SemanticOpenAICache(OpenAICache):
    def __init__(self, _openai_client, _semantic_cache):
        super().__init__(_openai_client)
        self.semantic_cache = _semantic_cache

    def generate(self, prompt):
        if prompt not in self.cache:
            similar_doc = self.semantic_cache.query(query_texts=[prompt], n_results=1)  # vecDB에서 임베딩 벡터로 검색

            # 검색 결과가 존재하고, 거리가 충분히(예시는 0.2 미만) 가까우면 문서 반환
            if len(similar_doc['distance'][0]) > 0 and similar_doc['distance'][0][0] < 0.2:
                return similar_doc['metadatas'][0][0]['response']
            else:  # 아니라면 답변 생성
                response = self.openai_client.chat.completions.create(model='gpt-3.5-turbo',
                                                                      messages=[{
                                                                          'role': 'user',
                                                                          'content': prompt
                                                                      }],
                                                                      )
                self.cache[prompt] = self.response_text(response)
                # 유사 캐시용 vecDB에 프롬프트-응답 쌍 저장, 윗 줄은 일치 캐시
                self.semantic_cache.add(documents=[prompt], metadatas=[{"response": self.response_text(response)}],
                                        ids=[prompt])

            return self.cache[prompt]


def semantic_cache_run():
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    openai_ef = OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-ada-002"
    )

    semantic_cache = chroma_client.create_collection(name="semantic_cache",
                                                     embedding_function=openai_ef, metadata={"hnsw:space": "cosine"})

    openai_cache = SemanticOpenAICache(openai_client, semantic_cache)

    questions = ["북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?",
                 "북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?",
                 "북태평양 기단과 오호츠크해 기단이 만나 한반도에 머무르는 기간은?",
                 "국내에 북태평양 기단과 오호츠크해 기단이 함께 머무리는 기간은?"]
    for question in questions:
        start_time = time.time()
        response = openai_cache.generate(question)
        print(f'질문: {question}')
        print("소요 시간: {:.2f}s".format(time.time() - start_time))
        print(f'답변: {response}\n')
