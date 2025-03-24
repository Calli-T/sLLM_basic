from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch


# 모델과 토크나이저 로드
model_name = "VIRNECT/llama-3-Korean-8B-V3"  # 사용하고자 하는 모델 이름
model = AutoAWQForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 양자화
model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"})

# 저장
model.save_quantized("./temp")
tokenizer.save_pretrained("./temp")
'''
# 프롬프트 설정
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")

# 텍스트 생성
output_tokens = model.generate(**inputs)
generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# 결과 출력
print("Generated Text:", generated_text)

'''
