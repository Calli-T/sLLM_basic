import autogen
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import PIL
import requests
from openai import OpenAI
from PIL import Image

from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.img_utils import _to_pil, get_image_data
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent

'''
프롬프트 입력 -> 이미지 생성 -> 유사 이미지 생성 하는 멀티 모달 에이전트 제작
'''


# - configuration 장전 -
def get_multimodal_and_diff_model_configs():
    _config_list_4o = autogen.config_list_from_json(
        "./for_ignore/OAI_CONFIG_LIST.json",
        filter_dict={
            "model": ["gpt-4o"],
        },
    )

    _config_list_dalle = autogen.config_list_from_json(
        "./for_ignore/OAI_CONFIG_LIST.json",
        filter_dict={
            "model": ["dall-e-3"],
        },
    )

    return _config_list_4o, _config_list_dalle


def get_llm_config():
    config_list = autogen.config_list_from_json(
        "OAI_CONFIG_LIST.json",
        file_location='./for_ignore',
        filter_dict={
            "model": ["gpt-4-turbo-preview"],
        },
    )
    return {
        "config_list": config_list,
        "temperature": 0,
    }


# llm_config, diff_model_config = get_llm_and_diff_model_configs()

# - 달리 api 함수와 이를 활용하는 달리 에이전트 클래스 선언 -

from for_ignore.openai_api_key import get_api_key


def dalle_call(client, prompt, model="dall-e-3", size="1024x1024", quality="standard", n=1) -> str:
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=n,
    )
    image_url = response.data[0].url
    img_data = get_image_data(image_url)
    return img_data


class DALLEAgent(ConversableAgent):
    def __init__(self, name, llm_config: dict, **kwargs):
        super().__init__(name, llm_config=llm_config, **kwargs)

        try:
            config_list = llm_config["config_list"]
            api_key = config_list[0]["api_key"]
        except Exception as e:
            print("Unable to fetch API Key, because", e)
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.register_reply([Agent, None], DALLEAgent.generate_dalle_reply)

    def generate_dalle_reply(self, messages, sender, config):
        client = self.client if config is None else config
        if client is None:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]

        prompt = messages[-1]["content"]
        img_data = dalle_call(client=self.client, prompt=prompt)
        plt.imshow(_to_pil(img_data))
        plt.axis("off")
        plt.show()
        return True, 'result.jpg'


# - 이미지 생성 에이전트 선언 -

config_list_4o, config_list_dalle = get_multimodal_and_diff_model_configs()
llm_config = get_llm_config()

painter = DALLEAgent(name="Painter", llm_config={"config_list": config_list_dalle})

user_proxy = UserProxyAgent(
    name="User_proxy", system_message="A human admin.", human_input_mode="NEVER", max_consecutive_auto_reply=0
)

'''# - 이미지 생성 에이전트 실행하기 -
user_proxy.initiate_chat(
    painter,
    message="갈색의 털을 가진 귀여운 강아지를 그려줘",
)'''

# - image 멀티 모달 GPT-4o 에이전트 생성 -
# user proxy, groupchat, pm도 같이
image_agent = MultimodalConversableAgent(
    name="image-explainer",
    system_message="Explane input image for painter to create similar image.",
    max_consecutive_auto_reply=10,
    llm_config={"config_list": config_list_4o, "temperature": 0.5, "max_tokens": 1500},
)

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    code_execution_config=False
)

groupchat = autogen.GroupChat(agents=[user_proxy, image_agent, painter], messages=[], max_round=12)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)


# 유사 이미지 생성 작업 실행하기
# 순서는 이미지 보기 -> 설명하기 -> 설명 달리에 넣고 생성인데 이거 user_proxy가 init 걸고 groupchat안에서 설명가랑 화가가 알아서 처리하는 모양이다?
user_proxy.initiate_chat(
    manager,
    message=f"""아래 이미지랑 비슷한 이미지를 만들어줘.
<img https://th.bing.com/th/id/R.422068ce8af4e15b0634fe2540adea7a?rik=y4OcXBE%2fqutDOw&pid=ImgRaw&r=0>.""",
) # ※ 이거 생성 자체는 멀쩡히 되는데, 오류가 난다

'''# - 멀티 모달 에이전트에 텍스트로 명령 -
user_proxy.initiate_chat(
    manager,
    message="갈색의 털을 가진 귀여운 강아지를 그려줘",
)
'''