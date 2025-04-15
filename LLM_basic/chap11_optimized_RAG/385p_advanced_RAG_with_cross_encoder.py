# ----- 데이터셋 -----

# - 평가용 데이터셋 1000개만 선별해서 가져오기 -

from datasets import load_dataset


def get_test_dataset():
    klue_mrc_test = load_dataset('klue', 'mrc', split='validation', cache_dir='./for_ignore/dataset_temp_klue_mrc')
    klue_mrc_test = klue_mrc_test.train_test_split(test_size=1000, seed=42)['test']

    return klue_mrc_test


# ----- 로컬 벡터 검색 구현 -----

import faiss


def make_embedding_index(emb_model, corpus):
    '''
    임베딩 모델과 말뭉치를 받아 임베딩을 만들어 마치 로컬 DB처럼 동작하게 하는 함수
    :param emb_model: 임베딩 모델
    :param corpus: 말뭉치
    :return: 만들어낸 임베딩과 그것들을 검색할 수 있는 기능을 모두 포함한 인덱스
    '''
    embeddings = emb_model.encode(corpus)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def find_embedding_top_k(query, emb_model, index, k):
    '''
    쿼리와 이것 저것을 받아 인덱스에서 유사도 상위 k개의 벡터의 인덱스들을 반환하는 함수
    :param query: 질의문
    :param emb_model: 질의문을 임베딩으로 만들 모델
    :param index: 이미 만들어진 임베딩들의 DB
    :param k: 반환 값의 수를 제어
    :return: k개의 상위 유사도를 가진 벡터들의 인덱스들(의 list?)
    '''

    embedding = emb_model.encode(query)
    distances, indices = index.search(embedding, k)
    return indices


# ----- 교차 인코더를 활용한 (벡터 검색 상위 K개에 대한) 순위 재정렬 -----
import numpy as np


# 교차 인코더에 넣기위해, 쿼리 - 검색 결과물쌍을 (상위 K개 만큼) 만들어 반환
def make_question_context_pairs(question_idx, indices):
    global klue_mrc_test

    return [[klue_mrc_test['question'][question_idx], klue_mrc_test['context'][idx]] for idx in indices]


# 순위 재정렬
def rerank_top_k(cross_model, question_idx, indices):  # , k):
    input_examples = make_question_context_pairs(question_idx, indices)  # 쌍을 만들어서
    relevance_scores = cross_model.predict(input_examples)  # 유사도를 계산하고
    reranked_indices = indices[np.argsort(relevance_scores)[::-1]]  # 큰 값 순서대로 정렬한다
    return reranked_indices


# ----- 평가 함수, hit_rate -----
# hit_rate: 상위 k개 안에 정답이 있는 비율

import time
import numpy as np
from tqdm.auto import tqdm


# hit_rate, 걸리는 시간 반환 하는 함수
def evaluate_hit_rate(datasets, embedding_model, index, k=10):
    start_time = time.time()
    predictions = []
    for question in datasets['question']:
        predictions.append(find_embedding_top_k(question, embedding_model, index, k)[0])
    total_prediction_count = len(predictions)
    hit_count = 0
    questions = datasets['question']
    contexts = datasets['context']
    for idx, prediction in enumerate(predictions):
        for pred in prediction:
            if contexts[pred] == contexts[idx]:
                hit_count += 1
                break
    end_time = time.time()
    return hit_count / total_prediction_count, end_time - start_time


# hit_rate, 걸린 시간 반환하고 순위 재정렬 해준 결과도 추가로 반환해주는 함수
def evaluate_hit_rate_with_rerank(datasets, embedding_model, cross_model, index, bi_k=30, cross_k=10):
    start_time = time.time()
    predictions = []
    for question_idx, question in enumerate(tqdm(datasets['question'])):
        indices = find_embedding_top_k(question, embedding_model, index, bi_k)[0]
        predictions.append(rerank_top_k(cross_model, question_idx, indices, k=cross_k))
    total_prediction_count = len(predictions)
    hit_count = 0
    questions = datasets['question']
    contexts = datasets['context']
    for idx, prediction in enumerate(predictions):
        for pred in prediction:
            if contexts[pred] == contexts[idx]:
                hit_count += 1
                break
    end_time = time.time()
    return hit_count / total_prediction_count, end_time - start_time, predictions


# ----- 실 사용 -----
# - 데이터셋 준비 -
klue_mrc_test = get_test_dataset()

from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

# - sts로 사전 학습한 임베딩 모델의 검색 능력 평가 -
base_embedding_model = SentenceTransformer('./for_ignore/model_trained_sts_klue-roberta-base')
base_index = make_embedding_index(base_embedding_model, klue_mrc_test['context'])
print(evaluate_hit_rate(klue_mrc_test, base_embedding_model, base_index, 10))

# - 위 모델을 mrc로 미세 조정된 모델의 검색 능력 평가 -
finetuned_embedding_model = SentenceTransformer('./for_ignore/model_fine_tuned_with_mrc_mmr_klue-roberta-base')
finetuned_index = make_embedding_index(finetuned_embedding_model, klue_mrc_test['context'])
print(evaluate_hit_rate(klue_mrc_test, finetuned_embedding_model, finetuned_index, 10))

# - 크로스 인코더와 섞어서 성능 평가, top 30개를 뽑아서 상위 10개를 다시 집어온다 -
cross_model = CrossEncoder('./for_ignore/model_cross_encoder_fine_tuned_with_mrc_dataset', num_labels=1)
hit_rate, cosumed_time, predictions = evaluate_hit_rate_with_rerank(klue_mrc_test, finetuned_embedding_model,
                                                                    cross_model, finetuned_index, bi_k=30, cross_k=10)
print(hit_rate, cosumed_time)
