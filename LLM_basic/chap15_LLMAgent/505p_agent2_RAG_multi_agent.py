import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent


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


# ----- RAG 에이전트 -----
def get_agents(_llm_config):
    _assistant = RetrieveAssistantAgent(
        name="assistant",
        system_message="You are a helpful assitant.",
        llm_config=_llm_config
    )

    _ragproxyagent = RetrieveUserProxyAgent(
        name="ragproxyagent",
        retrieve_config={
            "task": "qa",
            "docs_path": "https://ko.wikipedia.org/wiki/%ED%94%BC%EC%95%84%EB%85%B8",
            # "https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
            "collection_name": "default-sentence-transformers"
        }
    )


    return _assistant, _ragproxyagent


# assistant, ragproxyagent = get_agents(get_llm_config())
#
# assistant.reset()
# ragproxyagent.initiate_chat(assistant, problem="피아노의 종류는 어떤게 있어?")  # problem="AutoGen이 뭐야?")

'''
# 외부 정보를 쓰지 못하는 기본 에이전트
assistant.reset()
userproxyagent = autogen.UserProxyAgent(
    name="userproxyagent",
)
userproxyagent.initiate_chat(assistant, message="Autogen이 뭐야?")
'''

# - 임베딩 모델 커스텀, (예시는 openai 임베딩 모델로) -
from chromadb.utils import embedding_functions
from for_ignore.openai_api_key import get_api_key


def get_customed_embedding_rag_agents():
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=get_api_key(),
        model_name='text-embedding-3-small'
    )

    _ragproxyagent = RetrieveUserProxyAgent(
        name="ragproxyagent",
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER", retrieve_config={
            "task": "qa",
            "docs_path": "https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
            "embedding_function": openai_ef,
            "collection_name": "default-sentence-transformers"
        }
    )

    return _ragproxyagent


# assistant, _ = get_agents(get_llm_config())
# assistant.reset()
# ragproxyagent = get_customed_embedding_rag_agents()
# ragproxyagent.initiate_chat(assistant, problem="AutoGen이 뭐야?")

# ----- multi agent -----

from autogen import AssistantAgent


def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


# - 멀티 에이전츠들을 선언 -
def get_multi_agents(_llm_config):
    # RAG 안쓰는 사용자 역할 (proxy)
    _user = autogen.UserProxyAgent(
        name="Admin",
        is_termination_msg=termination_msg,
        human_input_mode="NEVER",
        system_message="The boss who ask questions and give tasks.",
        code_execution_config=False,
        default_auto_reply="Reply `TERMINATE` if the task is done.",
    )

    # RAG를 쓰는 사용자 역할 (proxy) 에이전트
    _user_rag = RetrieveUserProxyAgent(
        name="Admin_RAG",
        is_termination_msg=termination_msg,
        system_message="Assistant who has extra content retrieval power for solving difficult problems.",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        code_execution_config=False,
        retrieve_config={
            "task": "code",
            "docs_path": "https://microsoft.github.io/autogen/dev/user-guide/autogenstudio-user-guide/index.html",
            # "https://raw.githubusercontent.com/microsoft/autogen/main/samples/apps/autogen-studio/README.md",
            "chunk_token_size": 1000,
            "collection_name": "groupchat-rag",
        }
    )

    # 아래 어시들은 모두 role이 부여되었고, LLM으로 된 에이전트이며, 끝나면 TERMINATE를 외치라고 프롬프트를 보내놨음
    # 프로그래머 역할의 어시스트 에이전트
    _coder = AssistantAgent(
        name="Senior_Python_Engineer",
        is_termination_msg=termination_msg,
        system_message="You are a senior python engineer. Reply `TERMINATE` in the end when everything is done.",
        llm_config=_llm_config,
    )

    # 프로덕트 매니저 역할의 어시스트 에이전트
    _pm = autogen.AssistantAgent(
        name="Product_Manager",
        is_termination_msg=termination_msg,
        system_message="You are a product manager. Reply `TERMINATE` in the end when everything is done.",
        llm_config=_llm_config,
    )

    return _user, _user_rag, _coder, _pm


# - 실행용 함수 -

def _reset_agents(_user, _user_rag, _coder, _pm):
    _user.reset()
    _user_rag.reset()
    _coder.reset()
    _pm.reset()


def rag_chat(_user, _user_rag, _coder, _pm, _PROBLEM, _llm_config):
    _reset_agents(_user, _user_rag, _coder, _pm)
    groupchat = autogen.GroupChat(
        agents=[_user_rag, _coder, _pm],
        messages=[], max_round=12, speaker_selection_method="round_robin"  # 라운드 로빈? cpu 스케줄러에 나오던 그거랑 똑같은 방식으로 조율하는건가
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=_llm_config)

    _user_rag.initiate_chat(
        manager,
        problem=_PROBLEM,
    )


def norag_chat(_user, _user_rag, _coder, _pm, _PROBLEM, _llm_config):
    _reset_agents(_user, _user_rag, _coder, _pm)
    groupchat = autogen.GroupChat(
        agents=[_user, _coder, _pm],
        messages=[],
        max_round=12,
        speaker_selection_method="auto",
        allow_repeat_speaker=False,
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=_llm_config)

    _user.initiate_chat(
        manager,
        message=_PROBLEM,
    )


# llm_config = get_llm_config()
# user_proxy, user_rag_proxy, coder_assist, pm_assist = get_multi_agents(llm_config)
# PROBLEM = "AutoGen Studio는 무엇이고 AutoGen Studio로 어떤 제품을 만들 수 있을까?"
# norag_chat(user_proxy, user_rag_proxy, coder_assist, pm_assist, PROBLEM, llm_config)
# rag_chat(user_proxy, user_rag_proxy, coder_assist, pm_assist, PROBLEM, llm_config)
