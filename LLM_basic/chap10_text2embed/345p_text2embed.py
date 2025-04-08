from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', cache_folder='./model_temp')

embs = model.encode(['잠이 안 옵니다',
                     '졸음이 옵니다',
                     '기차가 옵니다'])

cos_scores = util.cos_sim(embs, embs)
print(cos_scores)

'''
# 명시적으로 바이 인코더 생성하는 예제 코드
from sentence_transformers import SentenceTransformer, models

def get_bi_encoder():
    # 사용할 BERT 모델
    word_embedding_model = models.Transformer('klue/roberta-base')
    # 풀링 층 차원 입력하기
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    # 두 모듈 결합하기
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    return model
'''

'''
# 평균 모드
# 모델의 마지막 출력을에(BERT는 층이 여러개!), 패딩을 거르고, 평균을 구한다
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask
'''
