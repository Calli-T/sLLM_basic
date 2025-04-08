from PIL import Image
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('clip-ViT-B-32', cache_folder='./model_temp_clip')

img_embs = model.encode([Image.open('dog.jpg'), Image.open('cat.jpg')])
text_embs = model.encode(['A dog on grass', 'Brown cat on yellow background'])

cos_scores = util.cos_sim(img_embs, text_embs)
print(cos_scores)