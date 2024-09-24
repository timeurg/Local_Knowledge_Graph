from .api import API


def get_short_title(content):
    api = API(
        options={
            "num_ctx": 50
    })
    messages = [
        {"role": "system", "content": "You are a concise summarizer. Provide a very short title (under 20 characters) for the given content."},
        {"role": "user", "content": f"Summarize this in under 20 characters: \n{content}"}
    ]
    
    # title_data = ""
    # for chunk in stream_api_call(messages, 50):
    #     title_data += chunk

    response = api.chat(messages=messages)
    
    short_title = response['message']['content'].strip()[:20]

    return short_title