import json
from for_ignore.openai_api_key import get_api_key

openai_api_key = get_api_key()


# ----- config -----
# - config json 으로 저장 -
def save_config_json():
    with open("./for_ignore/OAI_CONFIG_LIST.json", 'w') as f:
        config_list = [
            {
                "model": "gpt-4-turbo-preview",
                "api_key": openai_api_key,
            },
            {
                "model": "gpt-4o",
                "api_key": openai_api_key,
            },
            {
                "model": "dall-e-3",
                "api_key": openai_api_key,
            }
        ]

        json.dump(config_list, f)


# save_config_json()

# - config json 불러오기 -
import autogen


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


llm_config = get_llm_config()

# ----- AutoGen의 두 핵심요소 -----
from autogen import AssistantAgent, UserProxyAgent


def get_assistant_agent_and_user_proxy_agent(_llm_config):
    '''
    user_proxy: 사용자의 역할을 대신하는 Agent
    assistant: 사용자의 요청을 처리하는 Agent


    :return:
    '''
    _assistant = AssistantAgent("assistant", llm_config=_llm_config)
    _user_proxy = UserProxyAgent("user_proxy", is_termination_msg=lambda x: x.get("content", "") and x.get("content",
                                                                                                           "").rstrip().endswith(
        "TERMINATE"), human_input_mode="NEVER", code_execution_config={"work_dir": "coding", "use_docker": False})
    # content의 값이 존재하고, TERMIATE로 메시지가 끝나면 대화나 작업을 종료한다는 소리, 나머지는 설정에 관한 내용

    return _assistant, _user_proxy


assistant, user_proxy = get_assistant_agent_and_user_proxy_agent(llm_config)
user_proxy.initiate_chat(assistant, message="""
삼성전자의 지난 12개월의 주식 가격 그래프를 그려서 samsung_stock_price.png 파일로 저장해줘.
matplotlib 라이브러리를 사용해서 그리는 코드를 짜줘.
값을 잘 확인할 수 있도록 y축은 구간 최솟값에서 시작하도록 해줘.
이미지 비율은 보기 좋게 적절히 설정해줘.
""")
# 12개월치 달라했는데 10일치 주고 땡이더라, 서비스 범위를 공식 축소했음, fetch를 여러번 하나? 일단 2개월치는 잘 그리긴 하더라
# 설치 오류난 것도 잘 처리해주고, 코드 짠다음 실행하라 그럼
# plotly 오류가 생겼음 <- 이거 브라우저에서만 돌아간다는데? matplotlib으로 바꿨음
# ※ 이 예제의 User Proxy Agent는 LLM을 쓰지 않고, 사용자 대리로 메시지 전달, 끝남 조건, 인터페이스, 실행 환경 조성등의 상호작용만을 한다.
# ※ assistant의 LLM에, 프롬프트에 role을지정해주는 것도 가능하다고 한다 (Gemini 2.5 Pro 피셜) 초기화 때 system_message 파라미터를 사용해서 가능하다고 함