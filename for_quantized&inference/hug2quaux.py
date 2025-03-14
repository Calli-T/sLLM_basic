from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import init_empty_weights
import os

# 사용자 지정 경로
model_path = "./quantized_model"
cache_dir = "./cache"  # 캐시를 저장할 디렉토리

# BitsAndBytesConfig를 사용하여 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype='float16',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# VRAM의 한계를 넘기 위해 init_empty_weights 사용
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        "MLP-KTLim/llama-3-Korean-Bllossom-8B",  # 모델 이름 또는 허깅페이스 허브 경로
        quantization_config=bnb_config,
        cache_dir=cache_dir  # cache_dir 인자 사용
    )

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    "MLP-KTLim/llama-3-Korean-Bllossom-8B",
    cache_dir=cache_dir  # 캐시 경로 동일하게 설정
)

# 모델과 토크나이저를 지정한 경로에 저장
os.makedirs(model_path, exist_ok=True)
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
