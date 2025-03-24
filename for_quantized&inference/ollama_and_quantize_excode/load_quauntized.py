from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 저장된 모델 및 토크나이저 로드
model_path = "./quantized_model"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 프롬프트 입력
prompt = "안녕하세요! 당신의 이름은 무엇인가요?"

# 토큰화 및 입력 데이터 생성
inputs = tokenizer(prompt, return_tensors="pt")

# 모델 추론
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

# 결과 출력
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
