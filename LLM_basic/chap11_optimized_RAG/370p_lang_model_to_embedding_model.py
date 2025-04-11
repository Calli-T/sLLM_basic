# ----- 임베딩 모델 아키텍쳐 -----
from sentence_transformers import SentenceTransformer, models


def get_embedding_model():
    # 10장과 다르게 이미 학습된 임베딩은 아니라서 풀링 레이어를 따로 붙이고, 학습도 해줘야함

    transformer_model = models.Transformer('klue/roberta-base', cache_dir='./for_ignore/model_temp')

    pooling_layer = models.Pooling(
        transformer_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )
    embedding_model = SentenceTransformer(modules=[transformer_model, pooling_layer])

    return embedding_model


# ----- 유사 문장 데이터셋으로 데이터로더  준비 -----

from datasets import load_dataset


# - 데이터셋 가져옴 -
def get_sts_dataset():
    klue_sts_train = load_dataset('klue', 'sts', split='train', cache_dir='./for_ignore/dataset_temp')
    klue_sts_test = load_dataset('klue', 'sts', split='validation', cache_dir='./for_ignore/dataset_temp')
    # print(klue_sts_train[0])

    # 학습 데이터셋의 10%를 검증 데이터셋으로 구성한다.
    # train과 test밖에없어서, eval을 따로 만들어줘야함
    klue_sts_train = klue_sts_train.train_test_split(test_size=0.1, seed=42)
    klue_sts_train, klue_sts_eval = klue_sts_train['train'], klue_sts_train['test']

    return klue_sts_train, klue_sts_eval, klue_sts_test


from sentence_transformers import InputExample  # 이거 s_t 입력 전용 클래스이고, text쌍 list랑 label에 점수 int, 이거 두 개면된다


def preprocess_sts(datasets):
    all_examples = []
    for dataset in datasets:
        examples = []
        for data in dataset:
            examples.append(
                InputExample(
                    texts=[data['sentence1'], data['sentence2']],
                    label=data['labels']['label'] / 5.0)  # 원본 데이터가 1부터 5까지다 점수가
            )
        all_examples.append(examples)

    return all_examples


train_datasets, eval_datasets, test_datasets = preprocess_sts(get_sts_dataset())
print(train_datasets[0])

# - 데이터로더에 배치/셔플로 장착

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_datasets, shuffle=True, batch_size=16)

# - 검증, 평가는 데이터로더가 아니라 다른 내부 클래스 사용
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

eval_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(eval_datasets)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_datasets)

# ----- 학습 ------
text2embed_model = get_embedding_model()

# - 학습 전 성능 점수 확인 -


for score_name, value in test_evaluator(text2embed_model).items():
    print(f'{score_name}: {value}')

# - 학습 -
from sentence_transformers import losses

num_epochs = 4
model_name = 'klue/roberta-base'
model_save_path = './for_ignore/model_trained_sts'  # + model_name.replace("/", "-")
train_loss = losses.CosineSimilarityLoss(model=text2embed_model)

# 임베딩 모델 학습
text2embed_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=eval_evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=100,
    output_path=model_save_path
)

# - 학습 후 평가 -
trained_embedding_model = SentenceTransformer(model_save_path)
test_evaluator(trained_embedding_model)
