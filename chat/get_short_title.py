from chat.api import stream_api_call


def get_short_title(content):
    messages = [
        {"role": "system", "content": "You are a concise summarizer. Provide a very short title (under 20 characters) for the given content."},
        {"role": "user", "content": f"Summarize this in under 20 characters: {content[:100]}..."}
    ]
    
    title_data = ""
    for chunk in stream_api_call(messages, 50):
        title_data += chunk
    
    short_title = title_data.strip()[:20]
    return short_title