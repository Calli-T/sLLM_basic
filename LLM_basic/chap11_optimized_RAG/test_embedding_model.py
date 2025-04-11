from sentence_transformers import SentenceTransformer, util

model_save_path = './for_ignore/model_trained_sts_klue-roberta-base'
emb_model = SentenceTransformer(model_save_path)

emb1 = emb_model.encode("안녕하세요")
emb2 = emb_model.encode("감사합니다")
emb3 = emb_model.encode("안녕")
emb4 = emb_model.encode("안녕, 세상아?")
emb5 = emb_model.encode("태초에 하나님이 천지를 창조하시니라")
emb6 = emb_model.encode("태초에 하나님이 천지를 창조하셨다")

similarity = util.cos_sim(emb1, emb2)
print(similarity)
similarity = util.cos_sim(emb1, emb3)
print(similarity)
similarity = util.cos_sim(emb4, emb3)
print(similarity)
similarity = util.cos_sim(emb5, emb6)
print(similarity)