import math
import numpy as np
from typing import List
from transformers import PreTrainedTokenizer
from collections import defaultdict


class BM25:
    def __init__(self, corpus: List[List[str]], tokenizer: PreTrainedTokenizer):
        '''

        :param corpus: 문서 전체이다
        :param tokenizer: AutoTokenizer.from_pretrained의 반환 형이다. 여러 구체적인 Tokenizer(BertTokenizer, GPT2Tokenizer 등)클래스들의 부모 클래스

        하는일
        1. tokenizer, corpus 등록받기
        2. 문서수 self에 넣기
        3. 모든 문서 토큰화
        4. 평균 문서 길이 계산
        5. 내부 함수에 이름 달아주기
        '''
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.tokenized_corpus = self.tokenizer(corpus, add_special_tokens=False)['input_ids']
        self.n_docs = len(self.tokenized_corpus)
        self.avg_doc_lens = sum(len(lst) for lst in self.tokenized_corpus) / len(self.tokenized_corpus)
        self.idf = self._calculate_idf()
        self.term_freqs = self._calculate_term_freqs()

    # IDF(q_i) = ln{(N - n(q_i) + 0.5)/(n(q_i) + 0.5) + 1}의 구현
    def _calculate_idf(self):
        idf = defaultdict(float)  # 단어 별 dict
        for doc in self.tokenized_corpus:
            for token_id in set(doc):  # 문서 하나당 한 단어가 두 번 안나오도록 set
                idf[token_id] += 1
        for token_id, doc_frequency in idf.items():  # 토큰 번호와 빈도를 가지고 수식에 집어 넣음
            idf[token_id] = math.log(((self.n_docs - doc_frequency + 0.5) / (doc_frequency + 0.5)) + 1)
        return idf

    # f(q_i, D)의 구현, 모든 문서 별로 단어 어떤 단어가 얼마나 나오나 dict로 만들어둠
    # [문서 번호][토큰 번호]
    # dict를 원소로 가진 list
    def _calculate_term_freqs(self):
        term_freqs = [defaultdict(int) for _ in range(self.n_docs)]
        for i, doc in enumerate(self.tokenized_corpus):
            for token_id in doc:
                term_freqs[i][token_id] += 1
        return term_freqs

    # 쿼리, k1, b를 받아서 문서 각각에 SCORE(Q, D) 계산해줌
    '''
    쿼리를 토큰화
    scores는 문서 각각에 대한 점수
    쿼리의 단어 각각에 대해 연산, 연산은 다음과 같음
    IDF(q_i)를 계산
    self._calculate_term_freqs의 반환 값은 term_freqs[문서 번호][토큰번호]로 등장 횟수를 기록해둔,
    dict을 원소로 가진 list에서
    문서별 인덱스에서 토큰의 등장 빈도f(q_i, D)를 구한다음
    식에다 넣고, 인덱스에 해당하는 문서에 점수를 추가함
    
    '''

    def get_scores(self, query: str, k1: float = 1.2, b: float = 0.75):
        query = self.tokenizer([query], add_special_tokens=False)['input_ids'][0]
        scores = np.zeros(self.n_docs)
        for q in query:
            idf = self.idf[q]
            for i, term_freq in enumerate(self.term_freqs):
                q_frequency = term_freq[q]
                doc_len = len(self.tokenized_corpus[i])
                score_q = idf * (q_frequency * (k1 + 1)) / (
                        (q_frequency) + k1 * (1 - b + b * (doc_len / self.avg_doc_lens)))
                scores[i] += score_q
        return scores

    # 점수 높은 순서대로 점수와 인덱스를 뽑아준다, 위의 get_scores 함수를 사용함
    def get_top_k(self, query: str, k: int):
        scores = self.get_scores(query)
        top_k_indices = np.argsort(scores)[-k:][::-1]
        top_k_scores = scores[top_k_indices]
        return top_k_scores, top_k_indices


# ----- BM25 사용례 -----
# 필요한 모든것을 준비
from transformers import AutoTokenizer
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss


def get_tokenizer_dataset_embmodel_index():
    # 토크나이저와 데이터셋과 임베딩 모델
    _tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base', cache_dir='./for_ignore/model_klue_temp')
    _klue_mrc_dataset = load_dataset('klue', 'mrc', split='train', cache_dir='./for_ignore/datasets_temp')
    _sentence_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS',
                                          cache_folder='./for_ignore/model_temp')  # - 1000개 선택하고 임베딩으로 변환 -
    _klue_mrc_dataset = _klue_mrc_dataset.train_test_split(train_size=1000, shuffle=False)['train']
    _embeddings = _sentence_model.encode(_klue_mrc_dataset['context'])
    _index = faiss.IndexFlatL2(_embeddings.shape[1])  # 인덱스 만들기, 첫 매개변수는 차원수를 지정
    _index.add(_embeddings)  # 인덱스에 임베딩 저장하기

    return _tokenizer, _klue_mrc_dataset, _index, _sentence_model


tokenizer, klue_mrc_dataset, index, sentence_model = get_tokenizer_dataset_embmodel_index()


def test_BM25():
    bm25 = BM25(['안녕하세요', '반갑습니다', '안녕 서울'], tokenizer)
    print(bm25.get_scores('안녕'))


# test_BM25()
# 1, 3문서는 안녕이 포함되어있는데, 2문서는 없으니 점수가 0

def test_BM25_v2():
    # BM25 검색 준비
    # 키워드가 없으니 헛소리하는 예제이다
    bm25 = BM25(klue_mrc_dataset['context'], tokenizer)

    query = "이번 연도에는 언제 비가 많이 올까?"
    _, bm25_search_ranking = bm25.get_top_k(query, 100)

    for idx in bm25_search_ranking[:3]:
        print(klue_mrc_dataset['context'][idx][:50])


# test_BM25_v2()

def test_BM25_v3():
    bm25 = BM25(klue_mrc_dataset['context'], tokenizer)
    query = klue_mrc_dataset[3]['question']  # 로버트 헨리 딕이 1946년에 매사추세츠 연구소에서 개발한 것은 무엇인가?
    _, bm25_search_ranking = bm25.get_top_k(query, 100)

    for idx in bm25_search_ranking[:3]:
        print(klue_mrc_dataset['context'][idx][:50])
    # 결국 키워드가 많이 일치하면 잘찾는다, 이 케이스는 본문에 키워드가 많이 나오는 경우


# test_BM25_v3()


# ------------------------------------------------------------
# ----- 상호 순위 조합, reciprocal_rank_fusion -----
# - 구현 -
from collections import defaultdict


def reciprocal_rank_fusion(rankings: List[List[int]], k=5):
    rrf = defaultdict(float)
    for ranking in rankings:
        for i, doc_id in enumerate(ranking, 1):
            rrf[doc_id] += 1.0 / (k + i)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)


def RRF_test1():
    rankings = [[1, 4, 3, 5, 6], [2, 1, 3, 6, 4]]
    print(reciprocal_rank_fusion(rankings))


# RRF_test1()

# ------------------------------------------------------------
# ----- Hybrid Search -----
# - 구현 -

def dense_vector_search(query: str, k: int):
    # 쿼리의 임베딩과 일치율이 높은 임베딩과의 거리와 원본 문장의 인덱스 k개 반환
    query_embedding = sentence_model.encode([query])
    distances, indices = index.search(query_embedding, k)

    return distances[0], indices[0]


def hybrid_search(query, k=20):
    bm25 = BM25(klue_mrc_dataset['context'], tokenizer)

    _, dense_search_ranking = dense_vector_search(query, 100)
    _, bm25_search_ranking = bm25.get_top_k(query, 100)

    results = reciprocal_rank_fusion([dense_search_ranking, bm25_search_ranking], k=k)
    return results


# - 사용 -


query = "이번 연도에는 언제 비가 많이 올까?"
print("검색 쿼리 문장: ", query)
results = hybrid_search(query)
for idx, score in results[:3]:
    print(klue_mrc_dataset['context'][idx][:50])

print("=" * 80)
query = klue_mrc_dataset[3]['question']  # 로버트 헨리 딕이 1946년에 매사추세츠 연구소에서 개발한 것은 무엇인가?
print("검색 쿼리 문장: ", query)

results = hybrid_search(query)
for idx, score in results[:3]:
    print(klue_mrc_dataset['context'][idx][:50])
