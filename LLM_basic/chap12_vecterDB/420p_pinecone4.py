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


# pc, client = get_pinecone_openai_client(openai_api_key.get_api_key(), pinecone_api_key.get_api_key())

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
