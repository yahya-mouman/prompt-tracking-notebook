from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.utils import Prompt
from ddtrace import patch
from openai import OpenAI
import openai
from dotenv import load_dotenv
import os

load_dotenv()
patch(openai=True)

openai.api_key = os.environ.get("openai_api_key")
client = OpenAI()
load_dotenv()

LLMObs.enable(
    api_key=os.environ.get("dd_api_key"),
    site=os.environ.get("dd_site"),
    ml_app="prompt-tracking-sandbox-v2",
    service="prompt-tracking-sandbox-v2",
    agentless_enabled=True,
    env="testing",
    integrations_enabled=False,
)

prompt_id = "prompt-tracking-sandbox-prompt"
prompt_name = "recommendation_engine"


def query_openai(system_prompt:str=None, user_prompt:str=None, variables=[], model="gpt-4o", max_tokens=100, temperature=0.7):
    """ Sends a prompt to the OpenAI API and returns the text response. """
    if system_prompt and user_prompt:
        system_prompt_rendered = system_prompt.format(**variables)
        user_prompt_rendered = user_prompt.format(**variables)
        completion = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{
                "role": "system",
                "content": system_prompt_rendered
            },
                {
                    "role": "user",
                    "content": user_prompt_rendered
                }]
        )
    elif user_prompt:
        user_prompt_rendered = user_prompt.format(**variables)
        completion = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{
                "role": "user",
                "content": user_prompt_rendered
            }]
        )
    elif system_prompt:
        system_prompt_rendered = system_prompt.format(**variables)
        completion = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{
                "role": "system",
                "content": system_prompt_rendered
            }]
        )
    return completion.choices[0].message.content

prompt_step1 = "{user_prompt}"
variables_step1 = {"user_prompt": "Suggest a destination and a restaurant."}

with LLMObs.annotation_context(prompt=Prompt(id=prompt_id,
                           name=prompt_name,
                           version="1.0.0",
                           chat_template=[{"role":"User", "content":prompt_step1}],
                           variables=variables_step1
                           )):
    response_step1 = query_openai(user_prompt=prompt_step1, variables=variables_step1)

print("=== USER Prompt Template (Step 1) ===")
print(prompt_step1)
print("\n=== Variables (Step 1) ===")
print(variables_step1)
print("\n=== Response (Step 1) ===")
print(response_step1)

prompt_step2 = """
    {user_prompt}
    Examples:
        {examples}
"""

# Examples
examples_step2 = """
Example 1:
Input: "Bob: Can you suggest a restaurant?"
Output: "Based on your preferences for Italian cuisine, I recommend trying Luigi's Trattoria."

Example 2:
Input: "Alice: What's a good place for nature walks?"
Output: "Considering your love for scenic views, I suggest visiting Green Park Gardens."
"""

variables_step2 = {
    "user_prompt": "Suggest a destination and a restaurant.",
    "examples": examples_step2
    }

with LLMObs.annotation_context(prompt=Prompt(id=prompt_id,
                           name=prompt_name,
                           version="1.1.0",
                           chat_template=[{"role":"User", "content":prompt_step2}],
                           variables=variables_step2
                           )):
    response_step2 = query_openai(user_prompt=prompt_step2, variables=variables_step2)

print("=== USER Prompt template (Step 2) ===")
print(prompt_step2)
print("\n=== Variables (Step 2) ===")
print(variables_step2)
print("\n=== Response (Step 2) ===")
print(response_step2)

system_prompt_step3 = """
            You are a recommendation bot with a focus on user preferences. Your main task is to provide the best recommendations based on provided user preferences.
            Here are some rules and examples:

            - Always base your responses on user preferences.
            - Do not provide answers beyond the scope of the context given.

            Example exchanges:
            {examples}

            Use these examples to format your responses. 

            {constraints}

            Now, here's your specific task:
    """

prompt_step3 = "{user_prompt}"

examples_step3 = """
Example 1:
Input: "Bob: Can you suggest a restaurant?"
Output: "Based on your preferences for Italian cuisine, I recommend trying Luigi's Trattoria."

Example 2:
Input: "Alice: What's a good place for nature walks?"
Output: "Considering your love for scenic views, I suggest visiting Green Park Gardens."
"""

variables_step3 = {
    "user_prompt": "Suggest a destination and a restaurant.",
    "examples": examples_step3,
    "constraints": "Do not ask the user for more information. Either provide a recommendation or state that data is insufficient for a recommendation."
    }

with LLMObs.annotation_context(prompt=Prompt(id=prompt_id,
                           name=prompt_name,
                           version="2.0.0",
                           chat_template=[{"role":"System", "content":system_prompt_step3},{"role":"User", "content":prompt_step3}],
                           variables=variables_step3
                           )):
    response_step3 = query_openai(system_prompt=system_prompt_step3, user_prompt=prompt_step3, variables=variables_step3)

print("=== SYSTEM Prompt template (Step 3) ===")
print(system_prompt_step3)
print("=== USER Prompt template (Step 3) ===")
print(prompt_step3)
print("\n=== Variables (Step 3) ===")
print(variables_step3)
print("\n=== Response (Step 3) ===")
print(response_step3)

system_prompt_step4 = """
        You are a recommendation bot with a focus on user preferences. Your main task is to provide the best recommendations based on provided user preferences.
        Here are some rules and examples:

        - Always base your responses on user preferences.
        - Do not provide answers beyond the scope of the context given.

        Example exchanges:
        {examples}

        Knowing the users' preferences is key to providing accurate recommendations. Here is the context for the current user:
        {name}'s Context: {context}

        Use these examples to format your responses. 

        {constraints}

        Now, here's your specific task:
    """

prompt_step4 = "{user_prompt}"

examples_step4 = """
Example 1:
Input: "Bob: Can you suggest a restaurant?"
Output: "Based on your preferences for Italian cuisine, I recommend trying Luigi's Trattoria."

Example 2:
Input: "Alice: What's a good place for nature walks?"
Output: "Considering your love for scenic views, I suggest visiting Green Park Gardens."
"""

variables_step4 = {
    "name": "Alice",
    "context": "Alice is allergic to peanuts and enjoys traveling to beaches and mountains. She prefers vegetarian food.",
    "user_prompt": "Suggest a destination and a restaurant.",
    "examples": examples_step4,
    "constraints": "Do not ask the user for more information. Either provide a recommendation or state that data is insufficient for a recommendation."
    }

with LLMObs.annotation_context(prompt=Prompt(id=prompt_id,
                           name=prompt_name,
                           version="2.1.0",
                           chat_template=[{"role":"System", "content":system_prompt_step4},{"role":"User", "content":prompt_step4}],
                           variables=variables_step4
                           )):
    response_step4 = query_openai(system_prompt=system_prompt_step4, user_prompt=prompt_step4, variables=variables_step4)

print("=== SYSTEM Prompt template (Step 4) ===")
print(system_prompt_step4)
print("=== USER Prompt template (Step 4) ===")
print(prompt_step4)
print("\n=== Variables (Step 4) ===")
print(variables_step4)
print("\n=== Response (Step 4) ===")
print(response_step4)