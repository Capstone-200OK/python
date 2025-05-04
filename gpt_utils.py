# gpt_utils.py
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def gpt_generate_text_single_prompt(system_prompt, user_prompt, model="gpt-4.1", max_tokens=300):
    """
    Calls GPT using the new 'client.chat.completions.create' style.
    """

    client = OpenAI(api_key = api_key)

    # Compose messages in the new format
    # 아래 예시에서는 "developer" 역할에 system_prompt를,
    # "user" 역할에 user_prompt를 넣는 식으로 구성했습니다.
    # 필요에 따라 role을 "system" / "user" 등으로 변경 가능합니다.
    messages = [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
    )

    # The GPT response is in completion.choices[0].message.content
    return completion.choices[0].message.content
