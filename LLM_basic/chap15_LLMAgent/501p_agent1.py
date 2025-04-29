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

# # - config json -
# import autogen
#
# config_list = autogen.get_config_list(
#     "OAI_CONFIG_LIST.json",
#     file_location='./for_ignore',
#     filter_dict={
#         "model": ["gpt-4-turbo-preview"],
#     },
# )
