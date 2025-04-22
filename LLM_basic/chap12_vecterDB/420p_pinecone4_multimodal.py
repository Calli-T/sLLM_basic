'''
파인콘으로 멀티 모달 검색을 해보자

이미지를 임베딩
이미지를 글로 만드는데는 CLIP
글을 다시 이미지로 만드는데는 달리 3
DIffusionDB 2M의 선두 1000개의 데이터 사용

데이터셋 준비(diffusionDB 1000개)
base64 인코딩 함수와 GPT-4o 요청 함수 작성
'''
# - 파인콘과 openAI 클라이언트 준비 -
import os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from for_ignore import openai_api_key, pinecone_api_key


def get_pinecone_openai_client(pc_key, open_key):
    _pc = Pinecone(api_key=pc_key)
    os.environ["OPENAI_API_KEY"] = open_key
    _client = OpenAI()

    return _pc, _client


# - 데이터셋 준비 -
from datasets import load_dataset
from matplotlib import pyplot as plt


def get_dataset():
    _dataset = load_dataset("poloclub/diffusiondb", "2m_first_1k", split='train',
                            cache_dir='./for_ignore/dataset_diffusion_db')

    # - example -
    '''example_index = 867

    original_prompt = _dataset[example_index]['prompt']
    print(original_prompt)

    original_image = _dataset[example_index]['image']
    plt.imshow(original_image)
    plt.show()'''

    return _dataset


def get_image_prompt(dataset, example_index):
    return dataset[example_index]['image'], dataset[example_index]['prompt']


# - GPT 4o 요청 함수와 인코딩 함수 -

import requests
import base64
from io import BytesIO


def make_base64(image):  # base64 인코딩
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def generate_description_from_image_gpt4(prompt, image64, openai_api_key):  # 이미지 -> 묘사해달라고 GPT-4o에 요청
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    response_oai = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    result = response_oai.json()['choices'][0]['message']['content']
    return result


# - 이미지 설명 생성 테스트 해보기-
def descript_test():
    original_image, original_prompt = get_image_prompt(get_dataset(), 867)
    image_base64 = make_base64(original_image)
    described_result = generate_description_from_image_gpt4("Describe provided image", image_base64,
                                                            openai_api_key.get_api_key())
    print(described_result)


# - 클라이언트 준비 & 인덱스 생성 -

def get_index(pinecone):
    print(pinecone.list_indexes())

    index_name = "llm-multimodal"
    try:
        pinecone.create_index(
            name=index_name,
            dimension=512,
            metric="cosine",
            spec=ServerlessSpec(
                "aws", "us-east-1"
            )
        )
        print(pinecone.list_indexes())
    except:
        print("Index already exists")
    idx = pinecone.Index(index_name)

    return idx


# pc, client = get_pinecone_openai_client(pinecone_api_key.get_api_key(), openai_api_key.get_api_key())
# index = get_index(pc)

# - (DB에서 이미 있던, 생성 될 때 당시의) 프롬프트 텍스트를 임베딩 벡터로 변환하고 파인콘 인덱스에 저장 -
import torch
from tqdm.auto import trange
from transformers import AutoTokenizer, CLIPTextModelWithProjection

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_image_embeddings(_dataset):
    text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32",
                                                             cache_dir='./for_ignore/image_embedding_model')
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32",
                                              cache_dir='./for_ignore/image_embedding_model')

    tokens = tokenizer(_dataset['prompt'], padding=True, return_tensors="pt", truncation=True)
    batch_size = 16
    text_embs = []
    for start_idx in trange(0, len(_dataset), batch_size):
        with torch.no_grad():
            outputs = text_model(input_ids=tokens['input_ids'][start_idx:start_idx + batch_size],
                                 attention_mask=tokens['attention_mask'][start_idx:start_idx + batch_size])
            text_emb_tmp = outputs.text_embeds
        text_embs.append(text_emb_tmp)
    text_embs = torch.cat(text_embs, dim=0)
    print(text_embs.shape)  # (1000, 512)

    return text_embs


def pinecone_create(_dataset, _text_embs, _index):
    input_data = []
    for id_int, emb, prompt in zip(range(0, len(_dataset)), _text_embs.tolist(), _dataset['prompt']):
        input_data.append(
            {
                "id": str(id_int),
                "values": emb,
                "metadata": {
                    "prompt": prompt
                }
            }
        )

    _index.upsert(
        vectors=input_data
    )


'''pc, client = get_pinecone_openai_client(pinecone_api_key.get_api_key(), openai_api_key.get_api_key())
index = get_index(pc)
dataset = get_dataset()
text_embs = get_image_embeddings(dataset)
pinecone_create(dataset, text_embs, index)'''

# - 이미지 임베딩을 사용해서 프롬프트 검색 -
from transformers import AutoProcessor, CLIPVisionModelWithProjection


def get_image_search_results(_original_image, _index, top_k=3):
    vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = processor(images=_original_image, return_tensors="pt")

    outputs = vision_model(**inputs)
    image_embeds = outputs.image_embeds

    results = _index.query(
        vector=image_embeds[0].tolist(),
        top_k=top_k,
        include_values=False,
        include_metadata=True
    )
    idx = int(results['matches'][0]['id'])

    return idx, results


'''pc, client = get_pinecone_openai_client(pinecone_api_key.get_api_key(), openai_api_key.get_api_key())
index = get_index(pc)
dataset = get_dataset()
original_image, _ = get_image_prompt(dataset, 867)
_, search_results = get_image_search_results(original_image, index)
print(search_results)
'''

# ----- 프롬프트를 달리에 넣어 이미지 생성하는 함수 정의 -----
from PIL import Image


def generate_image_dalle3(_client, _prompt):
    response_oai = _client.images.generate(
        model="dall-e-3",
        prompt=str(_prompt),
        size="1024x1024",
        quality="standard",
        n=1,
    )
    result = response_oai.data[0].url
    return result


def get_generated_image(image_url):
    generated_image = requests.get(image_url).content
    image_filename = 'gen_img.png'
    with open(image_filename, "wb") as image_file:
        image_file.write(generated_image)
    return Image.open(image_filename)


# ----- 이미지 생성 -----
'''
사전 작업들, 
순서는 api_key 가져오기, 클라 생성, 인덱스 지정,
데이터셋에서 이미지 하나와 프롬프트 같이 가져오기,
인코딩, gpt-4o에 넣고 묘사 가져오기
'''
p_key = pinecone_api_key.get_api_key()
o_key = openai_api_key.get_api_key()
pc, client = get_pinecone_openai_client(p_key, o_key)
index = get_index(pc)
dataset = get_dataset()
original_image, original_prompt = get_image_prompt(dataset, 867)
searched_idx, search_results = get_image_search_results(original_image, index)

image_base64 = make_base64(original_image)
described_result = generate_description_from_image_gpt4("Describe provided image", image_base64, o_key)

# - GPT-4o가 이미지를 보고 만든 프롬프트로 이미지 생성 -
gpt_described_image_url = generate_image_dalle3(client, described_result)
gpt4o_prompt_image = get_generated_image(gpt_described_image_url)

# - 원본 프롬프트로 이미지 생성 -
original_prompt_image_url = generate_image_dalle3(client, described_result)
original_prompt_image = get_generated_image(original_prompt_image_url)

# - 이미지 임베딩으로 검색한 유사 프롬프트로 이미지 생성 -
searched_prompt_image_url = generate_image_dalle3(client, dataset[searched_idx]['prompt'])
searched_prompt_image = get_generated_image(searched_prompt_image_url)

plt.imshow(original_image)
plt.show()
print(original_prompt_image_url)
print(gpt_described_image_url)
print(searched_prompt_image_url)

# ----- 이미지 출력은 거르고 url로 봐라 -----
# lazy-loading된걸 뭐 바로 load해야한다는데, 귀찮으니 냅두자
'''import matplotlib.pyplot as plt

images = [original_image, gpt4o_prompt_image, original_prompt_image, searched_prompt_image]
titles = ['(a)', '(b)', '(c)', '(d)']

fig, axes = plt.subplots(1, len(images), figsize=(15, 5))

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title)

plt.tight_layout()
plt.show()
'''