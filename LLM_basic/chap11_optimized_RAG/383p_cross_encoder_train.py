# ----- 교차 인코더 불러오기 -----
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_cross_encoder():
    model = CrossEncoder('klue/roberta-small', num_labels=1)

    '''AutoModelForSequenceClassification.from_pretrained('klue/roberta-small',
                                                       cache_dir='./for_ignore/model_cross_encoder_fine_tuned_with_mrc_dataset')
    AutoTokenizer.from_pretrained('klue/roberta-small', cache_dir='./for_ignore/model_cross_encoder_fine_tuned_with_mrc_dataset')
    model = CrossEncoder(
        './for_ignore/model_cross_encoder_fine_tuned_with_mrc_dataset/models--klue--roberta-small/snapshots/5fe1f0cb3946f0ea1c01e657cd1688771cf47802',
        num_labels=1)'''

    import shutil
    import os

    # HuggingFace 캐시 디렉토리 기본 경로
    cache_dir = os.path.expanduser("~/.cache/huggingface")

    # 전체 삭제
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"HuggingFace 캐시 삭제 완료: {cache_dir}")
    else:
        print("HuggingFace 캐시 디렉토리가 존재하지 않습니다.")

    return model


# ----- 데이터셋과 전처리 -----
from datasets import load_dataset
from sentence_transformers import InputExample


def get_datasets():
    klue_mrc_train = load_dataset('klue', 'mrc', split='train', cache_dir='./for_ignore/dataset_temp_klue_mrc')
    klue_mrc_test = load_dataset('klue', 'mrc', split='validation', cache_dir='./for_ignore/dataset_temp_klue_mrc')

    return klue_mrc_train, klue_mrc_test


def filter_datasets(datasets):
    new_datasets = []
    for dataset in datasets:
        dataset = dataset.to_pandas()
        dataset = dataset[['title', 'question', 'context']]  # 질의-관련내용과 negative pair를 만들 기준이 될 title만 유지

        new_datasets.append(dataset)

    return new_datasets


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


def preprocess_for_test_dataset(testset):
    examples = []
    for idx, row in testset[:100].iterrows():
        examples.append(
            InputExample(texts=[row['question'], row['context']], label=1)
        )
        examples.append(
            InputExample(texts=[row['question'], row['irrelevant_context']], label=0)
        )

    return examples


def preprocess_for_train_dataset(trainset):
    train_samples = []
    for idx, row in trainset.iterrows():
        train_samples.append(InputExample(
            texts=[row['question'], row['context']], label=1
        ))
        train_samples.append(InputExample(
            texts=[row['question'], row['irrelevant_context']], label=0
        ))

    return train_samples


# ----- 성능 평가 함수 -----
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator


def evaluate(model, examples):
    ce_evaluator = CECorrelationEvaluator.from_input_examples(examples)
    print(ce_evaluator(model))


# ----- 학습 함수 -----
from torch.utils.data import DataLoader


def fine_tune_cross_encoder(cross_model, train_samples):
    train_batch_size = 16
    num_epochs = 1
    model_save_path = './for_ignore/model_cross_encoder_fine_tuned_with_mrc_dataset'

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

    cross_model.fit(
        train_dataloader=train_dataloader,
        epochs=num_epochs,
        warmup_steps=100,
        output_path=model_save_path
    )


# ----- 실행 구간 -----
# - 모델 불러오기 -
cross_encoder = get_cross_encoder()

# - 데이터셋 불러오고 전처리 -
train_dataset, test_dataset = get_datasets()
train_dataset, test_dataset = filter_datasets([train_dataset, test_dataset])
train_dataset, test_dataset = gen_negative_pair([train_dataset, test_dataset])
test_dataset = preprocess_for_test_dataset(test_dataset)
train_dataset = preprocess_for_train_dataset(train_dataset)

# - 학습 전 성능 평가 -
evaluate(cross_encoder, test_dataset)

# - 학습 -
fine_tune_cross_encoder(cross_encoder, train_dataset)

# - 학습 후 성능 평가 -
evaluate(cross_encoder, test_dataset)
