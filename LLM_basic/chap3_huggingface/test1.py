''' 제목 -> 카테고리 예측 모델 '''
from datasets import load_dataset

# ----- preproces -----

# 뉴스 데이터셋
klue_tc_train = load_dataset('klue', 'ynat', split='train')
klue_tc_eval = load_dataset('klue', 'ynat', split='validation')

print(klue_tc_train)

# 불필요한 칼럼 제거
klue_tc_train = klue_tc_train.remove_columns(['guid', 'url', 'date'])
klue_tc_eval = klue_tc_eval.remove_columns(['guid', 'url', 'date'])
print(klue_tc_train)

# 라벨 작업
print(klue_tc_train.features['label'].names)
print(klue_tc_train.features['label'].int2str(1))
klue_tc_label = klue_tc_train.features['label']


# 데이터에 칼럼 추가
def make_str_label(batch):
    batch['label_str'] = klue_tc_label.int2str(batch['label'])
    return batch


# 사상함수로 칼럼 전부 추가
klue_tc_train = klue_tc_train.map(make_str_label, batched=True, batch_size=1000)
print(klue_tc_train[0])

# 학습/검증/테스트 데이터셋 분할
# 'train'데이터의 1만개를 test로 분리한다음, 그걸 train으로 가져오는 일종의 오용?을활용해서 1만개만 가져오고있다
# 'test'데이터는 같은 방법으로 천 개씩 test를 test로, train을 eval로 가져오는중
train_dataset = klue_tc_train.train_test_split(test_size=10000, shuffle=True, seed=42)['test']
dataset = klue_tc_eval.train_test_split(test_size=1000, shuffle=True, seed=42)
test_dataset = dataset['test']
valid_dataset = dataset['train'].train_test_split(test_size=1000, shuffle=True, seed=42)['test']

# ----- train -----

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW
)
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader


def tokenize_function(examples):  # 제목(title) 컬럼에 대한 토큰화
    return tokenizer(examples["title"], padding="max_length", truncation=True)


# 모델과 토크나이저 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "klue/roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                                           num_labels=len(train_dataset.features['label'].names))
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.to(device)


def make_dataloader(dataset, batch_size, shuffle=True):
    dataset = dataset.map(tokenize_function, batched=True).with_format("torch")  # 데이터셋에 토큰화 수행
    dataset = dataset.rename_column("label", "labels")  # 컬럼 이름 변경
    dataset = dataset.remove_columns(column_names=['title'])  # 불필요한 컬럼 제거
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# 데이터로더 만들기
train_dataloader = make_dataloader(train_dataset, batch_size=8, shuffle=True)
valid_dataloader = make_dataloader(valid_dataset, batch_size=8, shuffle=False)
test_dataloader = make_dataloader(test_dataset, batch_size=8, shuffle=False)


def train_epoch(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)  # 모델에 입력할 토큰 아이디
        attention_mask = batch['attention_mask'].to(device)  # 모델에 입력할 어텐션 마스크
        labels = batch['labels'].to(device)  # 모델에 입력할 레이블
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)  # 모델 계산
        loss = outputs.loss  # 손실
        loss.backward()  # 역전파
        optimizer.step()  # 모델 업데이트
        total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss


import numpy as np


def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    return avg_loss, accuracy

num_epochs = 1
optimizer = AdamW(model.parameters(), lr=5e-5)

# 학습 루프
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss = train_epoch(model, train_dataloader, optimizer)
    print(f"Training loss: {train_loss}")
    valid_loss, valid_accuracy = evaluate(model, valid_dataloader)
    print(f"Validation loss: {valid_loss}")
    print(f"Validation accuracy: {valid_accuracy}")

# Testing
_, test_accuracy = evaluate(model, test_dataloader)
print(f"Test accuracy: {test_accuracy}") # 정확도 0.82

# 모델의 예측 아이디와 문자열 레이블을 연결할 데이터를 모델 config에 저장
id2label = {i: label for i, label in enumerate(train_dataset.features['label'].names)}
label2id = {label: i for i, label in id2label.items()}
model.config.id2label = id2label
model.config.label2id = label2id

from huggingface_hub import login

login(token="본인의 허깅페이스 토큰 입력")
repo_id = f"본인의 아이디 입력/roberta-base-klue-ynat-classification"
# Trainer를 사용한 경우
# trainer.push_to_hub(repo_id)
# 직접 학습한 경우
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)