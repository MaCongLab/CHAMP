"""LLM API access. Importing this module does not create a client or send a request."""
import os


def make_client(api_key=None):
    from openai import OpenAI

    key = api_key or os.environ.get('OPENAI_API_KEY')
    if not key:
        raise ValueError('Export OPENAI_API_KEY before running annotation')
    return OpenAI(api_key=key)


def GPT_QA(prompt, model_name='gpt-5-mini', api_key=None, input=None, client=None):
    client = client or make_client(api_key)
    response = client.responses.create(model=model_name, instructions=prompt, input=input)
    return response.output_text
