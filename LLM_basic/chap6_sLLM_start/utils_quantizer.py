from transformers import AutoTokenizer, AutoModelForCausalLM
import bitsandbytes as bnb
import torch

# 모델과 토크나이저 로드
model_id = "beomi/Yi-Ko-6B"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='./model_path')
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='./model_path', device_map="cpu")


# 레이어별로 양자화 수행
def quantize_model_layers(model):
    for name, module in model.named_modules():
        print(name)
        if isinstance(module, torch.nn.Linear):
            quantized_layer = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None
            )
            quantized_layer.load_state_dict(module.state_dict())
            setattr(model, name, quantized_layer)
    print('quauntized')
    return model


quantized_model = quantize_model_layers(model)

# 양자화된 모델 저장
save_dir = "./quantized_model"
quantized_model.save_pretrained(save_dir)

# 토크나이저 저장
tokenizer.save_pretrained(save_dir)

print(f"Quantized model and tokenizer saved to {save_dir}")
