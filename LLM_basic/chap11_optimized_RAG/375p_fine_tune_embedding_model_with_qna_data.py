# ----- 사전 학습된 임베딩 모델 불러오기 -----
from sentence_transformers import SentenceTransformer


def get_pretrained_embedding_model():
    return SentenceTransformer('./for_ignore/model_trained_sts_klue-roberta-base')


# ----- 데이터셋 준비 -----
# 질문과 답변, 그리고 기사(내용, context)가 있다
# 질문과, 답이 담긴 내용끼리는 유사도 점수를 높게
# 대조 학습적으로, 관려이 없는 질문-내용 쌍에는 낮은 유사도 점수를
# 대조 학습으로 미세 조정 할 때, 도메인과 관련된 데이터를 사용하는 것이 추천된다.

from datasets import load_dataset


def get_datasets():
    klue_mrc_train = load_dataset('klue', 'mrc', split='train', cache_dir='./for_ignore/dataset_temp_klue_mrc')
    klue_mrc_test = load_dataset('klue', 'mrc', split='validation', cache_dir='./for_ignore/dataset_temp_klue_mrc')
    # for name, value in klue_mrc_train[0].items():
    #     print(f'{name}: {value}')

    return klue_mrc_train, klue_mrc_test


# - 전처리 -

# 필요한 값만 남김
def filter_datasets(datasets):
    new_datasets = []
    for dataset in datasets:
        dataset = dataset.to_pandas()
        dataset = dataset[['title', 'question', 'context']]  # 질의-관련내용과 negative pair를 만들 기준이 될 title만 유지

        new_datasets.append(dataset)

    return new_datasets


# 부정-쌍 생성(긍정-쌍은 이미 존재함)
# trainset 한정으로, 뻘짓이다. MNR 손실 함수로 자동으로 만들 수 있다
def gen_negative_pair(datasets):
    new_datasets = []
    for dataset in datasets:
        irrelevant_contexts = []
        for idx, row in dataset.iterrows():
            title = row['title']
            irrelevant_contexts.append(dataset.query(f"title != '{title}'").sample(n=1)['context'].values[0])
        dataset['irrelevant_context'] = irrelevant_contexts

        new_datasets.append(dataset)

    return new_datasets


# - 성능 평가용 데이터셋 클래스 생성(sentence transformer 내부 절차) -
from sentence_transformers import InputExample


def preprocess_for_test_dataset(testset):
    examples = []
    for idx, row in testset[:100].iterrows():  # 긍정-쌍은 라벨 1로, 부정-쌍은 라벨 0으로
        examples.append(
            InputExample(texts=[row['question'], row['context']], label=1)
        )
        examples.append(
            InputExample(texts=[row['question'], row['irrelevant_context']], label=0)
        )

    return examples


from sentence_transformers import datasets


def get_train_dataloader(trainset):
    # MNR 함수는, 배치 안의 다른 것은 모두 negative로 가정한다
    # 즉, 위에서 만든 trainset의 negative-pair는 뻘짓이다
    train_samples = []
    for idx, row in trainset.iterrows():
        train_samples.append(InputExample(
            texts=[row['question'], row['context']]
        ))

    return datasets.NoDuplicatesDataLoader(train_samples, batch_size=8)


# ----- 평가 함수 -----
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


def evaluate_model(model, examples):
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(examples)
    print(evaluator(model))


# ----- 손실 함수 -----
from sentence_transformers import losses


def get_MNR_loss(model):
    return losses.MultipleNegativesRankingLoss(model)


# ----- 실행 구간 -----
# - 모델 불러오기 -
sentence_model = get_pretrained_embedding_model()

# - 데이터셋 불러오고 전처리 -
train_dataset, test_dataset = get_datasets()
train_dataset, test_dataset = filter_datasets([train_dataset, test_dataset])
train_dataset, test_dataset = gen_negative_pair([train_dataset, test_dataset])
test_dataset = preprocess_for_test_dataset(test_dataset)
train_dataloader = get_train_dataloader(train_dataset)

# - 학습 전 성능 평가 -
evaluate_model(sentence_model, test_dataset)

# - 손실 함수 불러오기 -
loss = get_MNR_loss(sentence_model)

# - 학습 -
epochs = 1
save_path = './for_ignore/model_fine_tuned_with_mrc_mmr_klue-roberta-base'

sentence_model.fit(
    train_objectives=[(train_dataloader, loss)],
    epochs=epochs,
    warmup_steps=100,
    output_path=save_path,
    show_progress_bar=True
)

# - 미세 조정 후 성능 평가 -
evaluate_model(sentence_model, test_dataset)
