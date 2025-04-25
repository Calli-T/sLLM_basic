from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir="./for_ignore/clip_model")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir="./for_ignore/clip_model")

image = Image.open('./for_ignore/mint/2.jpg')

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
print(probs)
